//! lemonade-burn-server: OpenAI-compatible server for the Burn + ternary BitNet runtime.
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (blocking and
//! SSE), `POST /v1/completions` (blocking and SSE), `GET /api/v1/topology` (single-node
//! description, so the lemonade bridge's topology proxy has something to return),
//! `GET /v1/placement` (where expert calls went), `GET`/`POST /v1/tunables` (every knob the
//! engine reads; the live ones change without a restart, see `tunables.rs`).
//!
//! The model runs on one dedicated worker thread (it uses the rayon pool internally);
//! requests are served in arrival order. A client disconnect stops its generation.
//!
//! Launch from the lemonade bridge with a command template such as
//! `exec lemonade-burn-server --model {model} --port {port}`.

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::tensor::Device;
use clap::Parser;
use futures_util::stream::{self, Stream};
use lb_model::sample::{Sampler, SamplingParams};
use lb_model::text::{ChatMessage, ChatStyle, StreamDecoder, TextCodec};
use lb_model::BitNet;
use burn::tensor::Tensor;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

pub const RUNTIME_NAME: &str = "lemonade-burn";

pub mod tunables;
use tunables::{LiveTargets, Tunables};

#[derive(Parser, Debug, Clone)]
#[command(name = "lemonade-burn-server")]
struct Args {
    /// Model: a converted BitNet directory (lb-convert output) or an LFM2 / LFM2-MoE `.gguf`
    /// file (with `tokenizer.json` beside it).
    #[arg(long)]
    model: PathBuf,
    /// LFM2 only: directory of NautiBurn scrape files to plan the expert placement from.
    #[arg(long)]
    placement_from_scrapes: Option<PathBuf>,
    /// LFM2 only: GPU byte budget for resident experts (MB); 0 = all experts on the CPU.
    #[arg(long, default_value_t = 0)]
    gpu_budget_mb: usize,
    /// LFM2 only: upload the planned residents to the GPU named by LB_WGPU_DEVICE and run
    /// them there (needs the `gpu` feature).
    #[arg(long, default_value_t = false)]
    gpu: bool,
    /// LFM2 only: what to do with an expert the router asked for that is not resident.
    /// `cpu` runs it on the CPU reference path; `on-demand` copies it to the device and
    /// runs it there, which measured 45.2 ms against about 4 ms a call on the 24B.
    #[arg(long, default_value = "cpu")]
    miss_policy: String,
    /// LFM2 only, with `--gpu`: byte budget (MB) for resident *dense* weights — the
    /// short-conv projections, attention q/k/v/o and the leading dense FFNs. These run for
    /// every token, so there is nothing to plan: give it enough and all of them go over
    /// (424 MB on the 24B). 0 = leave the dense side on the CPU.
    #[arg(long, default_value_t = 0)]
    gpu_dense_mb: usize,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model id reported by /v1/models (default: model directory name).
    #[arg(long)]
    model_id: Option<String>,
    /// Max context (prompt + generated) tokens.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Compute threads (default: rayon default = logical CPUs, or RAYON_NUM_THREADS).
    #[arg(long, env = "LB_THREADS")]
    threads: Option<usize>,
    /// Default max_tokens when a request gives none.
    #[arg(long, default_value_t = 512)]
    default_max_tokens: usize,
}

struct Job {
    prompt: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
    stop: Vec<String>,
    tx: mpsc::UnboundedSender<Gen>,
}

#[derive(Debug)]
enum Gen {
    Text(String),
    Done { finish_reason: &'static str, prompt_tokens: usize, completion_tokens: usize, prefill_s: f64, decode_s: f64 },
    Error(String),
}

struct AppState {
    model_id: String,
    ctx: usize,
    /// Live (`/v1/tunables`): `max_tokens` for a request that gives none.
    default_max_tokens: Arc<AtomicUsize>,
    codec: Arc<TextCodec>,
    jobs: std_mpsc::Sender<Job>,
    started: u64,
    kernel: &'static str,
    threads: usize,
    /// LFM2 only: the live expert-placement counters, shared with the model thread.
    placement: Option<Arc<lb_lfm2::moe::PlacementStats>>,
    /// LFM2 only: the forward-pass time split, shared with the model thread.
    timings: Option<Arc<lb_lfm2::model::Timings>>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A decoder the server can host: prefill/decode with its own cache.
pub trait Decoder: Send + 'static {
    type Cache;
    fn new_cache(&self) -> Self::Cache;
    /// Logits `[1, vocab]` for the last of `tokens`, run at positions `cache.len..`.
    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache, all_logits: bool) -> Tensor<2>;
    fn max_context(&self) -> usize;
    fn bos(&self) -> u32;
    fn kernel_name(&self) -> &'static str;
    /// Called after each request (scrape flush, profiling); default nothing.
    fn after_request(&self, _prompt: &[u32]) {}
}

impl Decoder for BitNet {
    type Cache = lb_model::model::KvCache;
    fn new_cache(&self) -> Self::Cache {
        BitNet::new_cache(self)
    }
    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache, all_logits: bool) -> Tensor<2> {
        BitNet::forward(self, tokens, cache, all_logits)
    }
    fn max_context(&self) -> usize {
        self.cfg.max_position_embeddings
    }
    fn bos(&self) -> u32 {
        self.cfg.bos_token_id
    }
    fn kernel_name(&self) -> &'static str {
        lb_ternary::Kernel::detect().name()
    }
}

impl Decoder for lb_lfm2::Lfm2 {
    type Cache = lb_lfm2::Cache;
    fn new_cache(&self) -> Self::Cache {
        lb_lfm2::Lfm2::new_cache(self)
    }
    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache, all_logits: bool) -> Tensor<2> {
        lb_lfm2::Lfm2::forward(self, tokens, cache, all_logits)
    }
    fn max_context(&self) -> usize {
        self.cfg().n_ctx
    }
    fn bos(&self) -> u32 {
        self.cfg().bos
    }
    fn kernel_name(&self) -> &'static str {
        if self.store.n_resident() > 0 {
            "lfm2 q4_0 (resident experts on GPU, rest CPU)"
        } else {
            "lfm2 q4_0 (CPU)"
        }
    }
    fn after_request(&self, prompt: &[u32]) {
        if let Err(e) = self.scraper.flush("prefill", prompt) {
            eprintln!("scrape flush failed: {e}");
        }
    }
}

fn worker<D: Decoder>(model: D, codec: Arc<TextCodec>, rx: std_mpsc::Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(&model, &codec, &job)));
        if let Err(e) = res {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            let _ = job.tx.send(Gen::Error(format!("generation panicked: {msg}")));
        }
    }
}

fn run_job<D: Decoder>(model: &D, codec: &TextCodec, job: &Job) {
    let to_vec = |t: burn::tensor::Tensor<2>| t.into_data().try_to_vec::<f32>().expect("logits");
    let mut cache = model.new_cache();
    let mut sampler = Sampler::new(job.params.clone());
    let t0 = Instant::now();
    let logits = to_vec(model.forward(&job.prompt, &mut cache, false));
    let prefill_s = t0.elapsed().as_secs_f64();
    let mut next = sampler.sample(&logits);
    let mut dec = StreamDecoder::new();
    let mut text = String::new();
    let mut sent = 0usize; // bytes of `text` already sent
    let mut n = 0;
    let t1 = Instant::now();
    let mut finish = "length";
    let max_stop = job.stop.iter().map(|s| s.len()).max().unwrap_or(0);
    while n < job.max_tokens {
        if codec.is_stop(next) {
            finish = "stop";
            break;
        }
        n += 1;
        match dec.push(codec, next) {
            Ok(piece) => text.push_str(&piece),
            Err(e) => {
                let _ = job.tx.send(Gen::Error(e.to_string()));
                return;
            }
        }
        if let Some(pos) = job.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
            text.truncate(pos);
            if text.len() > sent {
                let _ = job.tx.send(Gen::Text(text[sent..].to_string()));
            }
            sent = text.len();
            finish = "stop";
            break;
        }
        // hold back enough text that a stop string split across tokens is not leaked
        let mut safe = text.len().saturating_sub(max_stop.saturating_sub(1));
        while !text.is_char_boundary(safe) {
            safe -= 1;
        }
        if safe > sent {
            if job.tx.send(Gen::Text(text[sent..safe].to_string())).is_err() {
                return; // client went away
            }
            sent = safe;
        }
        if n >= job.max_tokens {
            break;
        }
        let l = to_vec(model.forward(&[next], &mut cache, false));
        next = sampler.sample(&l);
    }
    if text.len() > sent {
        let _ = job.tx.send(Gen::Text(text[sent..].to_string()));
    }
    model.after_request(&job.prompt);
    let _ = job.tx.send(Gen::Done {
        finish_reason: finish,
        prompt_tokens: job.prompt.len(),
        completion_tokens: n,
        prefill_s,
        decode_s: t1.elapsed().as_secs_f64(),
    });
}

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<Value>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

fn message_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        // OpenAI content parts: concatenate the text parts
        Value::Array(parts) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

impl AppState {
    fn submit(&self, req: &ChatReq, prompt: Vec<u32>) -> Result<(mpsc::UnboundedReceiver<Gen>, usize), Response> {
        if prompt.len() >= self.ctx {
            return Err(err(StatusCode::BAD_REQUEST, format!("prompt has {} tokens; context is {}", prompt.len(), self.ctx)));
        }
        let max_tokens = req.max_tokens.unwrap_or_else(|| self.default_max_tokens.load(Ordering::Relaxed)).min(self.ctx - prompt.len());
        let stop = match &req.stop {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => vec![],
        };
        let params = SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(0),
            seed: req.seed.unwrap_or_else(|| now() ^ 0x9e37_79b9),
        };
        let (tx, rx) = mpsc::unbounded_channel();
        let n = prompt.len();
        self.jobs
            .send(Job { prompt, max_tokens, params, stop, tx })
            .map_err(|_| err(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped"))?;
        Ok((rx, n))
    }
}

async fn chat(State(st): State<Arc<AppState>>, Json(req): Json<ChatReq>) -> Response {
    let msgs: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage { role: m.get("role").and_then(Value::as_str).unwrap_or("user").to_string(), content: message_text(m.get("content").unwrap_or(&Value::Null)) })
        .collect();
    if msgs.is_empty() {
        return err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let prompt = match st.codec.encode_chat(&msgs) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
    };
    respond(st, req, prompt, true).await
}

async fn completions(State(st): State<Arc<AppState>>, Json(req): Json<ChatReq>) -> Response {
    let text = match &req.prompt {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 => a[0].as_str().unwrap_or("").to_string(),
        _ => return err(StatusCode::BAD_REQUEST, "prompt must be a string"),
    };
    let prompt = match st.codec.encode(&text) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
    };
    respond(st, req, prompt, false).await
}

async fn respond(st: Arc<AppState>, req: ChatReq, prompt: Vec<u32>, chat: bool) -> Response {
    let (mut rx, _n) = match st.submit(&req, prompt) {
        Ok(v) => v,
        Err(r) => return r,
    };
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = format!("{}-{:x}{:04x}", if chat { "chatcmpl" } else { "cmpl" }, now(), SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let created = now();
    // echo the caller's model name (Lemonade routes by its own id) when given
    let model = req.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| st.model_id.clone());
    if !req.stream {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(Gen::Text(t)) => text.push_str(&t),
                Some(Gen::Done { finish_reason, prompt_tokens, completion_tokens, prefill_s, decode_s }) => {
                    let choice = if chat {
                        json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason})
                    } else {
                        json!({"index": 0, "text": text, "finish_reason": finish_reason})
                    };
                    return Json(json!({
                        "id": id, "object": if chat {"chat.completion"} else {"text_completion"}, "created": created, "model": model,
                        "choices": [choice],
                        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                        "timings": {"prompt_s": prefill_s, "predicted_s": decode_s,
                                    "prompt_per_second": prompt_tokens as f64 / prefill_s.max(1e-9),
                                    "predicted_per_second": completion_tokens as f64 / decode_s.max(1e-9)}
                    }))
                    .into_response();
                }
                Some(Gen::Error(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
                None => return err(StatusCode::INTERNAL_SERVER_ERROR, "worker dropped request"),
            }
        }
    }
    let first = if chat {
        Some(json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}))
    } else {
        None
    };
    let s = stream_events(rx, first, id, created, model, chat);
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

fn stream_events(
    rx: mpsc::UnboundedReceiver<Gen>,
    first: Option<Value>,
    id: String,
    created: u64,
    model: String,
    chat: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let chunk = move |delta: Value, finish: Value| -> Value {
        if chat {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        } else {
            json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish}]})
        }
    };
    let queue: VecDeque<Event> = first.map(|f| Event::default().data(f.to_string())).into_iter().collect();
    // state: receiver (None once finished), queued events, chunk builder
    stream::unfold((Some(rx), queue, chunk), |(mut rx, mut queue, chunk)| async move {
        loop {
            if let Some(ev) = queue.pop_front() {
                return Some((Ok(ev), (rx, queue, chunk)));
            }
            let r = rx.as_mut()?;
            match r.recv().await {
                Some(Gen::Text(t)) => queue.push_back(Event::default().data(chunk(json!({"content": t}), Value::Null).to_string())),
                Some(Gen::Done { finish_reason, prompt_tokens, completion_tokens, .. }) => {
                    let mut last = chunk(json!({}), json!(finish_reason));
                    last["usage"] = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens});
                    queue.push_back(Event::default().data(last.to_string()));
                    queue.push_back(Event::default().data("[DONE]"));
                    rx = None;
                }
                Some(Gen::Error(e)) => {
                    queue.push_back(Event::default().data(json!({"error": {"message": e}}).to_string()));
                    queue.push_back(Event::default().data("[DONE]"));
                    rx = None;
                }
                None => return None,
            }
        }
    })
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{
        "id": st.model_id, "object": "model", "created": st.started, "owned_by": RUNTIME_NAME,
        "context_length": st.ctx
    }]}))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"status": "ok", "runtime": RUNTIME_NAME, "version": env!("CARGO_PKG_VERSION"), "model": st.model_id, "model_loaded": st.model_id}))
}

/// Where the expert calls went since start: how many the GPU residents caught, how many
/// fell through to the CPU, and the wall time each side cost. Cumulative, not per request.
async fn placement_stats(State(st): State<Arc<AppState>>) -> Json<Value> {
    let Some(s) = st.placement.as_ref() else {
        return Json(json!({"placement": "none (not an LFM2 model)"}));
    };
    let t = s.snapshot();
    let mut v = json!({
        "schema": "lemonade-burn.placement.v1",
        "gpu_hits": t.gpu_hits, "cpu_calls": t.cpu_calls, "on_demand": t.on_demand,
        "gpu_hit_rate": t.hit_rate,
        "gpu_s": t.gpu_s, "cpu_s": t.cpu_s, "on_demand_s": t.on_demand_s
    });
    // Where a forward pass went, so "what is left to optimise" is read off the server
    // instead of inferred from the token rate.
    if let Some(tm) = st.timings.as_ref() {
        let r2 = |x: f64| (x * 100.0).round() / 100.0;
        for (prefix, s) in [("all", tm.snapshot()), ("decode", tm.decode())] {
            v[format!("{prefix}_passes")] = json!(s.passes);
            v[format!("{prefix}_forward_ms")] = json!(r2(s.forward_ms));
            v[format!("{prefix}_matmul_ms")] = json!(r2(s.matmul_ms));
            v[format!("{prefix}_other_ms")] = json!(r2(s.other_ms));
        }
    }
    Json(v)
}

async fn get_tunables(State(t): State<Arc<Tunables>>) -> Json<Value> {
    Json(t.describe())
}

/// Change live knobs: `{"name": value, ...}`. All or nothing; a restart-only knob answers
/// 409 with the flag to use, a bad value or unknown name 400, and in either case nothing moved.
async fn post_tunables(State(t): State<Arc<Tunables>>, Json(req): Json<Value>) -> Response {
    let Some(map) = req.as_object() else {
        return err(StatusCode::BAD_REQUEST, "body must be an object of {\"name\": value}");
    };
    if map.is_empty() {
        return err(StatusCode::BAD_REQUEST, "nothing to change (GET /v1/tunables lists the knobs)");
    }
    match t.apply(map) {
        Ok(v) => Json(v).into_response(),
        Err(r) => (StatusCode::from_u16(r.status()).unwrap_or(StatusCode::BAD_REQUEST), Json(json!({"error": {"message": r.message(), "type": "tunable_error"}}))).into_response(),
    }
}

/// The `/v1/tunables` routes, with their own state so they can be served (and tested) apart.
fn tunables_router(t: Arc<Tunables>) -> Router {
    Router::new().route("/v1/tunables", get(get_tunables).post(post_tunables)).with_state(t)
}

async fn topology(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "schema": "lemonade-burn.topology.v1",
        "runtime": RUNTIME_NAME,
        "nodes": [{"name": "local", "kind": "cpu", "kernel": st.kernel, "threads": st.threads, "layers": "all"}]
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }
    let t0 = Instant::now();
    let dev = Device::flex();
    let is_gguf = args.model.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false);
    let (jtx, jrx) = std_mpsc::channel::<Job>();
    let default_max_tokens = Arc::new(AtomicUsize::new(args.default_max_tokens));
    type Loaded = (Arc<TextCodec>, usize, String, &'static str, Option<Arc<lb_lfm2::moe::PlacementStats>>, Option<Arc<lb_lfm2::model::Timings>>, LiveTargets);
    let (codec, ctx, model_id, kernel, placement, timings, live): Loaded = if is_gguf {
        let mut model = lb_lfm2::Lfm2::load(&args.model, &dev)?;
        let cfg = model.cfg().clone();
        model.store.knobs.set_miss_policy(lb_lfm2::moe::MissPolicy::parse(&args.miss_policy).map_err(|e| anyhow::anyhow!(e))?);
        if let Some(dir) = &args.placement_from_scrapes {
            let scrapes = lb_lfm2::moe::read_scrape_dir(dir)?;
            let hist = lb_lfm2::moe::Histogram::from_scrapes("lfm2", cfg.n_layer, cfg.n_expert, &scrapes);
            let bytes: Vec<usize> = (0..cfg.n_layer).map(|l| model.w.expert_bytes(l)).collect();
            model.placement = lb_lfm2::moe::Placement::plan(&hist, &bytes, args.gpu_budget_mb * 1_000_000);
            eprintln!("placement: {} experts planned resident ({:.0} MB) from {} scrape files", model.placement.n_resident(), model.placement.resident_bytes as f64 / 1e6, scrapes.len());
            if args.gpu {
                let n = model.load_residents().map_err(|e| anyhow::anyhow!(e))?;
                eprintln!("uploaded {n} experts ({:.0} MB) to the GPU", model.store.bytes as f64 / 1e6);
            }
        }
        if args.gpu && args.gpu_dense_mb > 0 {
            #[cfg(feature = "gpu")]
            {
                let want = lb_lfm2::dense::DenseStore::candidate_bytes(&model.w);
                let n = model.load_dense(args.gpu_dense_mb * 1_000_000).map_err(|e| anyhow::anyhow!(e))?;
                eprintln!(
                    "dense: {n} matrices resident ({:.0} MB of {:.0} MB available)",
                    model.dense.bytes as f64 / 1e6,
                    want as f64 / 1e6
                );
            }
            #[cfg(not(feature = "gpu"))]
            anyhow::bail!("--gpu-dense-mb needs the `gpu` feature");
        }
        let dir = args.model.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        let codec = Arc::new(TextCodec::load_with(&dir.join("tokenizer.json"), cfg.bos, ChatStyle::ChatMl, vec![cfg.eos])?);
        let ctx = args.ctx.min(cfg.n_ctx);
        let model_id = args.model_id.clone().unwrap_or_else(|| args.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "lfm2".into()));
        let kernel = model.kernel_name();
        let stats = model.stats.clone();
        let tm = model.timings.clone();
        let live = LiveTargets { knobs: Some(model.knobs()), scraper: Some(model.scraper.clone()), default_max_tokens: default_max_tokens.clone(), ctx };
        let wc = codec.clone();
        std::thread::Builder::new().name("lb-model".into()).spawn(move || worker(model, wc, jrx))?;
        (codec, ctx, model_id, kernel, Some(stats), Some(tm), live)
    } else {
        let model = BitNet::load(&args.model, &dev)?;
        let codec = Arc::new(TextCodec::load(&args.model, model.cfg.bos_token_id)?);
        let ctx = args.ctx.min(model.cfg.max_position_embeddings);
        let model_id = args.model_id.clone().unwrap_or_else(|| args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "bitnet".into()));
        let kernel = model.kernel_name();
        let live = LiveTargets { knobs: None, scraper: None, default_max_tokens: default_max_tokens.clone(), ctx };
        let wc = codec.clone();
        std::thread::Builder::new().name("lb-model".into()).spawn(move || worker(model, wc, jrx))?;
        (codec, ctx, model_id, kernel, None, None, live)
    };
    let fixed = Tunables::fixed_from_start(
        &args.model.to_string_lossy(),
        args.gpu,
        args.gpu_budget_mb,
        args.gpu_dense_mb,
        args.placement_from_scrapes.as_deref().map(|p| p.to_str().unwrap_or("")),
        ctx,
        rayon::current_num_threads(),
        &model_id,
    );
    let tunables = Arc::new(Tunables::new(live, fixed));
    eprintln!(
        "lemonade-burn-server: loaded {} in {:.1}s (kernel {}, {} threads, ctx {ctx})",
        model_id,
        t0.elapsed().as_secs_f64(),
        kernel,
        rayon::current_num_threads()
    );
    let state = Arc::new(AppState {
        model_id,
        ctx,
        default_max_tokens,
        codec,
        jobs: jtx,
        started: now(),
        kernel,
        threads: rayon::current_num_threads(),
        placement,
        timings,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/placement", get(placement_stats))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/api/v1/topology", get(topology))
        .with_state(state)
        .merge(tunables_router(tunables));
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", args.host, args.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("lemonade-burn-server: listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                #[cfg(unix)]
                {
                    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("sigterm handler");
                    tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
                }
                #[cfg(not(unix))]
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

#[cfg(test)]
mod http_tests {
    //! The routes end to end through the router (no model, no listener).
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn router() -> (Router, Arc<lb_lfm2::Knobs>) {
        let knobs = Arc::new(lb_lfm2::Knobs::default());
        let scraper = Arc::new(lb_lfm2::moe::Scraper::new(None, "m".into(), "c".into()));
        let live = LiveTargets { knobs: Some(knobs.clone()), scraper: Some(scraper), default_max_tokens: Arc::new(AtomicUsize::new(512)), ctx: 2048 };
        let fixed = Tunables::fixed_from_start("/m/x.gguf", true, 14500, 900, None, 2048, 80, "x");
        (tunables_router(Arc::new(Tunables::new(live, fixed))), knobs)
    }

    async fn call(app: &Router, method: &str, body: Option<Value>) -> (u16, Value) {
        let req = match body {
            Some(b) => Request::builder().method(method).uri("/v1/tunables").header("content-type", "application/json").body(Body::from(b.to_string())).unwrap(),
            None => Request::builder().method(method).uri("/v1/tunables").body(Body::empty()).unwrap(),
        };
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn get_lists_and_post_changes_live_knobs() {
        let (app, knobs) = router();
        let (status, v) = call(&app, "GET", None).await;
        assert_eq!(status, 200);
        let names: Vec<&str> = v["tunables"].as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"miss_policy") && names.contains(&"gpu_budget_mb") && names.contains(&"batch_miss"), "{names:?}");

        let (status, v) = call(&app, "POST", Some(json!({"miss_policy": "on-demand", "batch_miss": false}))).await;
        assert_eq!(status, 200, "{v}");
        assert_eq!(v["applied"]["miss_policy"], "on-demand");
        assert_eq!(knobs.miss_policy(), lb_lfm2::moe::MissPolicy::OnDemand);
        assert!(!knobs.batch_miss());
        let (_, v) = call(&app, "GET", None).await;
        let e = v["tunables"].as_array().unwrap().iter().find(|e| e["name"] == "miss_policy").unwrap();
        assert_eq!(e["value"], "on-demand");
    }

    #[tokio::test]
    async fn post_refuses_restart_only_with_409_and_the_flag() {
        let (app, knobs) = router();
        let (status, v) = call(&app, "POST", Some(json!({"gpu_budget_mb": 3200, "miss_policy": "on-demand"}))).await;
        assert_eq!(status, 409, "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains("--gpu-budget-mb"), "{v}");
        assert_eq!(knobs.miss_policy(), lb_lfm2::moe::MissPolicy::Cpu, "nothing in a refused post is applied");
        let (status, _) = call(&app, "POST", Some(json!({"no_such_knob": 1}))).await;
        assert_eq!(status, 400);
        let (status, _) = call(&app, "POST", Some(json!({}))).await;
        assert_eq!(status, 400);
        let (status, _) = call(&app, "POST", Some(json!([1, 2]))).await;
        assert_eq!(status, 400);
    }
}

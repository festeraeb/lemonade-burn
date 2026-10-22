//! lemonade-burn-server: OpenAI-compatible server for the Burn + ternary BitNet runtime.
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (blocking and
//! SSE), `POST /v1/completions` (blocking and SSE), `GET /api/v1/topology` (single-node
//! description, so the lemonade bridge's topology proxy has something to return).
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
use lb_model::text::{ChatMessage, StreamDecoder, TextCodec};
use lb_model::BitNet;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

pub const RUNTIME_NAME: &str = "lemonade-burn";

#[derive(Parser, Debug, Clone)]
#[command(name = "lemonade-burn-server")]
struct Args {
    /// Converted model directory (lb-convert output).
    #[arg(long)]
    model: PathBuf,
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
    default_max_tokens: usize,
    codec: Arc<TextCodec>,
    jobs: std_mpsc::Sender<Job>,
    started: u64,
    kernel: &'static str,
    threads: usize,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn worker(model: BitNet, codec: Arc<TextCodec>, rx: std_mpsc::Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(&model, &codec, &job)));
        if let Err(e) = res {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            let _ = job.tx.send(Gen::Error(format!("generation panicked: {msg}")));
        }
    }
}

fn run_job(model: &BitNet, codec: &TextCodec, job: &Job) {
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
        if TextCodec::is_stop(next) {
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
        let max_tokens = req.max_tokens.unwrap_or(self.default_max_tokens).min(self.ctx - prompt.len());
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
    let model = st.model_id.clone();
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
    let model = BitNet::load(&args.model, &dev)?;
    let codec = Arc::new(TextCodec::load(&args.model, model.cfg.bos_token_id)?);
    let ctx = args.ctx.min(model.cfg.max_position_embeddings);
    let model_id = args.model_id.clone().unwrap_or_else(|| args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "bitnet".into()));
    eprintln!(
        "lemonade-burn-server: loaded {} in {:.1}s (kernel {}, {} threads, ctx {ctx})",
        model_id,
        t0.elapsed().as_secs_f64(),
        lb_ternary::Kernel::detect().name(),
        rayon::current_num_threads()
    );
    let (jtx, jrx) = std_mpsc::channel::<Job>();
    let wc = codec.clone();
    std::thread::Builder::new().name("lb-model".into()).spawn(move || worker(model, wc, jrx))?;
    let state = Arc::new(AppState {
        model_id,
        ctx,
        default_max_tokens: args.default_max_tokens,
        codec,
        jobs: jtx,
        started: now(),
        kernel: lb_ternary::Kernel::detect().name(),
        threads: rayon::current_num_threads(),
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/api/v1/topology", get(topology))
        .with_state(state);
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", args.host, args.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("lemonade-burn-server: listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("sigterm handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
            })
            .await?;
        Ok(())
    })
}

//! Benchmark result rows, rendered as a GitHub markdown table or as JSON lines.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BenchRow {
    pub host: String,
    pub kernel: String,
    pub threads: usize,
    pub prompt_tokens: usize,
    pub prefill_s: f64,
    pub gen_tokens: usize,
    pub decode_s: f64,
    pub rss_mb: f64,
}

fn rate(tokens: usize, secs: f64) -> f64 {
    if secs <= 0.0 { 0.0 } else { tokens as f64 / secs }
}

impl BenchRow {
    pub fn prefill_tps(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_s)
    }

    pub fn decode_tps(&self) -> f64 {
        rate(self.gen_tokens, self.decode_s)
    }
}

pub fn markdown_table(rows: &[BenchRow]) -> String {
    let mut s = String::from(
        "| host | kernel | threads | prompt tok | prefill tok/s | gen tok | decode tok/s | RSS MB |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        s += &format!(
            "| {} | {} | {} | {} | {:.2} | {} | {:.2} | {:.0} |\n",
            r.host, r.kernel, r.threads, r.prompt_tokens, r.prefill_tps(), r.gen_tokens, r.decode_tps(), r.rss_mb
        );
    }
    s
}

pub fn json_lines(rows: &[BenchRow]) -> String {
    rows.iter()
        .map(|r| {
            let mut v = serde_json::to_value(r).expect("BenchRow serializes");
            v["prefill_tps"] = r.prefill_tps().into();
            v["decode_tps"] = r.decode_tps().into();
            v.to_string() + "\n"
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(prefill_s: f64) -> BenchRow {
        BenchRow {
            host: "c3".into(),
            kernel: "avx2".into(),
            threads: 4,
            prompt_tokens: 64,
            prefill_s,
            gen_tokens: 32,
            decode_s: 4.2,
            rss_mb: 1860.4,
        }
    }

    #[test]
    fn rates() {
        let r = row(1.3);
        assert_eq!(format!("{:.2}", r.prefill_tps()), "49.23");
        assert_eq!(format!("{:.2}", r.decode_tps()), "7.62");
        assert_eq!(row(0.0).prefill_tps(), 0.0);
    }

    #[test]
    fn markdown_has_header_separator_and_rows() {
        let t = markdown_table(&[row(1.3), row(2.0)]);
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("| host | kernel |"));
        assert!(lines[2].contains("| 49.23 |"), "{}", lines[2]);
        assert!(lines[2].ends_with("| 1860 |"), "{}", lines[2]);
    }

    #[test]
    fn json_lines_parse_and_carry_rates() {
        let j = json_lines(&[row(1.3), row(2.0)]);
        let lines: Vec<&str> = j.lines().collect();
        assert_eq!(lines.len(), 2);
        for l in lines {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert!(v.get("decode_tps").is_some() && v.get("prefill_tps").is_some());
            assert_eq!(v["host"], "c3");
        }
    }
}

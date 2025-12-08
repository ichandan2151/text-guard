// app/dashboard/[id]/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { supabase } from "../../lib/supabaseClient";

type DocumentRow = {
  id: string;
  created_at: string;
  file_name: string | null;
  document_link: string | null;
  storage_path: string | null;
  project_name: string | null;
  total_chunks: number | null;
  risk_chunks: number | null;
  no_risk_chunks: number | null;
  risk_score: number | null;
};

export default function DocumentDetailsPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [doc, setDoc] = useState<DocumentRow | null>(null);
  const [loading, setLoading] = useState(true);

  const [aiText, setAiText] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  const id = params?.id;

  useEffect(() => {
    const fetchDocAndAnalysis = async () => {
      if (!id) return;

      // auth check
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) {
        router.push("/login");
        return;
      }

      // load document row
      const { data, error } = await supabase
        .from("documents")
        .select("*")
        .eq("id", id)
        .single();

      if (error) {
        console.error(error);
        setLoading(false);
        return;
      }

      const docData = data as DocumentRow;
      setDoc(docData);
      setLoading(false);

      // call Gemini
      try {
        setAiLoading(true);
        setAiError(null);

        const res = await fetch("/api/gemini-summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileName: docData.file_name,
            createdAt: docData.created_at,
            totalChunks: docData.total_chunks,
            riskChunks: docData.risk_chunks,
            noRiskChunks: docData.no_risk_chunks,
            riskScore: docData.risk_score,
          }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error || "Failed to fetch Gemini summary.");
        }

        const json = await res.json();
        setAiText(json.text as string);
      } catch (err) {
        console.error(err);
        setAiError(
          err instanceof Error ? err.message : "Failed to fetch Gemini summary."
        );
      } finally {
        setAiLoading(false);
      }
    };

    fetchDocAndAnalysis();
  }, [id, router]);

  return (
    <main
      style={{
        minHeight: "100vh",
        backgroundColor: "#020617",
        color: "#f9fafb",
        padding: "32px 16px",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <div style={{ width: "100%", maxWidth: 960 }}>
        <button
          onClick={() => router.push("/dashboard")}
          style={{
            marginBottom: 16,
            fontSize: 13,
            background: "none",
            border: "1px solid #1f2937",
            borderRadius: 999,
            padding: "6px 14px",
            cursor: "pointer",
            color: "#e5e7eb",
          }}
        >
          ← Back to dashboard
        </button>

        {loading && <p style={{ fontSize: 14 }}>Loading document…</p>}

        {!loading && !doc && (
          <p style={{ fontSize: 14 }}>Document not found.</p>
        )}

        {doc && (
          <section
            style={{
              borderRadius: 20,
              border: "1px solid #1f2937",
              padding: 24,
              background:
                "radial-gradient(circle at top, rgba(15,23,42,0.8), #020617 60%)",
              boxShadow: "0 24px 60px rgba(0,0,0,0.7)",
              display: "flex",
              flexDirection: "column",
              gap: 16,
            }}
          >
            {/* Header */}
            <header>
              <h1 style={{ margin: 0, fontSize: 22 }}>
                {doc.file_name || "Untitled document"}
              </h1>
              <p
                style={{
                  marginTop: 4,
                  fontSize: 13,
                  color: "#9ca3af",
                }}
              >
                Uploaded {new Date(doc.created_at).toLocaleString()}
              </p>
            </header>

            {/* Skeleton content (we'll replace later with your design) */}
            <div
              style={{
                padding: 16,
                borderRadius: 16,
                border: "1px dashed #1f2937",
                fontSize: 13,
                color: "#9ca3af",
              }}
            >
              <p style={{ marginTop: 0, marginBottom: 8 }}>
                Detailed report view coming soon.
              </p>
              <p style={{ margin: 0 }}>
                Here we’ll show per-chunk risk, policy hits, charts, and a richer
                document preview once the design is ready.
              </p>
            </div>

            {/* Numeric stats */}
            <div
              style={{
                display: "flex",
                gap: 16,
                flexWrap: "wrap",
                fontSize: 13,
              }}
            >
              <div>
                <div style={{ color: "#6b7280" }}>Chunks</div>
                <div style={{ fontWeight: 600 }}>
                  {doc.total_chunks ?? "—"}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Risk</div>
                <div style={{ fontWeight: 600, color: "#fbbf24" }}>
                  {doc.risk_chunks ?? "—"}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>No risk</div>
                <div style={{ fontWeight: 600, color: "#34d399" }}>
                  {doc.no_risk_chunks ?? "—"}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Risk score</div>
                <div style={{ fontWeight: 600 }}>
                  {doc.risk_score !== null && doc.risk_score !== undefined
                    ? `${(doc.risk_score * 100).toFixed(1)}%`
                    : "—"}
                </div>
              </div>
            </div>

            {/* 🔮 Gemini AI insight */}
            <section
              style={{
                marginTop: 8,
                padding: 16,
                borderRadius: 16,
                border: "1px solid #1f2937",
                background:
                  "radial-gradient(circle at top, rgba(15,23,42,0.7), #020617 70%)",
              }}
            >
              <h2
                style={{
                  marginTop: 0,
                  marginBottom: 8,
                  fontSize: 15,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                AI Risk Insight
                <span
                  style={{
                    fontSize: 11,
                    padding: "2px 8px",
                    borderRadius: 999,
                    border: "1px solid #1f2937",
                    color: "#9ca3af",
                  }}
                >
                  Gemini
                </span>
              </h2>

              {aiLoading && (
                <p style={{ fontSize: 13, color: "#9ca3af" }}>
                  Generating summary from Gemini…
                </p>
              )}

              {aiError && (
                <p style={{ fontSize: 13, color: "#f97373" }}>{aiError}</p>
              )}

              {!aiLoading && !aiError && aiText && (
                <p
                  style={{
                    fontSize: 13,
                    whiteSpace: "pre-wrap",
                    margin: 0,
                  }}
                >
                  {aiText}
                </p>
              )}

              {!aiLoading && !aiError && !aiText && (
                <p style={{ fontSize: 13, color: "#9ca3af", margin: 0 }}>
                  No AI summary available yet.
                </p>
              )}
            </section>

            {/* Link to original */}
            {doc.document_link && (
              <a
                href={doc.document_link}
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: 13, color: "#38bdf8" }}
              >
                View original source
              </a>
            )}
          </section>
        )}
      </div>
    </main>
  );
}

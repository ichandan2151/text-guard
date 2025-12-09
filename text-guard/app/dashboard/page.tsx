// app/dashboard/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "../lib/supabaseClient";

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
  processing_status: string | null;
  ai_insight: string | null;
  ai_insight_model: string | null;
  ai_insight_generated_at: string | null;
};

const PROCESSOR_URL = process.env.NEXT_PUBLIC_PROCESSOR_URL;

async function callProcessor({
  documentId,
  storagePath,
  fileName,
}: {
  documentId: string;
  storagePath: string;
  fileName: string;
}) {
  if (!PROCESSOR_URL) {
    console.error("NEXT_PUBLIC_PROCESSOR_URL is not set");
    throw new Error("Processor URL is not configured");
  }

  const url = `${PROCESSOR_URL.replace(/\/$/, "")}/process_document`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: documentId,
      storage_path: storagePath,
      file_name: fileName,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error("Processor error:", res.status, text);
    throw new Error(`Processor failed (${res.status})`);
  }

  return res.json();
}

function getFileExtension(fileName: string | null) {
  if (!fileName) return "";
  const parts = fileName.split(".");
  if (parts.length < 2) return "";
  return parts[parts.length - 1].toUpperCase();
}

export default function DashboardPage() {
  const router = useRouter();

  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const init = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) {
        router.push("/login");
        return;
      }

      await fetchDocuments();
    };

    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);

    const { data, error } = await supabase
      .from("documents")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error(error);
      setError("Failed to load documents.");
    } else {
      setDocs((data || []) as DocumentRow[]);
    }

    setLoading(false);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    router.push("/login");
  };

  const handleOpenFilePicker = () => {
    setUploadError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    try {
      await uploadFileToSupabase(file);
    } catch (err) {
      console.error(err);
      setUploadError(
        err instanceof Error ? err.message : "Upload failed."
      );
    }
  };

  const uploadFileToSupabase = async (file: File) => {
    try {
      setUploading(true);
      setUploadError(null);

      const filePath = `${Date.now()}-${file.name}`;

      const { data: storageData, error: storageError } = await supabase.storage
        .from("documents")
        .upload(filePath, file);

      if (storageError || !storageData) {
        console.error(storageError);
        throw new Error("Failed to upload file to storage.");
      }

      const { data: publicUrlData } = supabase.storage
        .from("documents")
        .getPublicUrl(storageData.path);

      const publicUrl = publicUrlData?.publicUrl ?? null;

      const { data: inserted, error: insertError } = await supabase
        .from("documents")
        .insert({
          file_name: file.name,
          storage_path: storageData.path,
          document_link: publicUrl,
          project_name: null,
          total_chunks: null,
          risk_chunks: null,
          no_risk_chunks: null,
          risk_score: null,
          processing_status: "processing",
        })
        .select("*")
        .limit(1);

      if (insertError) {
        console.error(insertError);
        throw new Error("File stored, but failed to save metadata.");
      }

      const docRow = inserted?.[0] as DocumentRow | undefined;

      if (docRow) {
        await callProcessor({
          documentId: docRow.id,
          storagePath: storageData.path,
          fileName: file.name,
        });
      }

      await fetchDocuments();

      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } finally {
      setUploading(false);
    }
  };

  const handleOpenDetails = (id: string) => {
    if (typeof window !== "undefined") {
      window.open(`/dashboard/${id}`, "_blank");
    }
  };

  return (
    <main
      style={{
        minHeight: "100vh",
        backgroundColor: "#020617",
        color: "#f9fafb",
        display: "flex",
        justifyContent: "center",
        padding: "40px 16px",
      }}
    >
      <div style={{ width: "100%", maxWidth: 1200 }}>
        {/* HERO / HEADER - centered, no 'View documents' */}
        <header
          style={{
            marginBottom: 32,
            paddingBottom: 24,
            borderBottom: "1px solid #111827",
          }}
        >
          {/* Logout row */}
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginBottom: 24,
            }}
          >
            <button
              onClick={handleLogout}
              style={{
                padding: "8px 16px",
                borderRadius: 999,
                border: "1px solid #374151",
                backgroundColor: "transparent",
                color: "#e5e7eb",
                fontSize: 13,
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              Log out
            </button>
          </div>

          {/* Centered hero content */}
          <div
            style={{
              maxWidth: 720,
              margin: "0 auto",
              textAlign: "center",
            }}
          >
            <h1
              style={{
                fontSize: 36,
                lineHeight: 1.1,
                margin: 0,
                fontWeight: 600,
              }}
            >
              Automated Risk &amp; Compliance Auditor
            </h1>
            <p
              style={{
                marginTop: 12,
                marginBottom: 16,
                fontSize: 14,
                color: "#9ca3af",
              }}
            >
              Assess project documentation for social, environmental, and governance risks. Upload files and examine TextGuard's AI-powered insights
            </p>

            <p
              style={{
                marginTop: 0,
                marginBottom: 20,
                fontSize: 13,
                color: "#9ca3af",
              }}
            >
              Upload your supervision reports, memos, or audits and check them
              against your internal risk and compliance policies.
            </p>

            {/* Upload control */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                id="file-input"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />

              <button
                type="button"
                onClick={handleOpenFilePicker}
                disabled={uploading}
                style={{
                  padding: "10px 22px",
                  borderRadius: 999,
                  border: "none",
                  background:
                    "linear-gradient(to right, #2563eb, #1d4ed8, #1d4ed8)",
                  color: "#f9fafb",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: uploading ? "default" : "pointer",
                  opacity: uploading ? 0.7 : 1,
                  minWidth: 180,
                }}
              >
                {uploading ? "Uploading…" : "Upload document"}
              </button>

              {selectedFile && !uploading && (
                <span style={{ fontSize: 12, color: "#9ca3af" }}>
                  Selected: {selectedFile.name}
                </span>
              )}

              {uploadError && (
                <span style={{ fontSize: 12, color: "#f97373" }}>
                  {uploadError}
                </span>
              )}
            </div>
          </div>
        </header>

        {/* Status / loading */}
        <div style={{ marginBottom: 12 }}>
          {loading && (
            <span style={{ fontSize: 12, color: "#9ca3af" }}>Loading…</span>
          )}
          {error && (
            <p style={{ fontSize: 13, color: "#f97373", marginTop: 4 }}>
              {error}
            </p>
          )}
        </div>

        {/* Cards grid */}
        {!loading && docs.length === 0 && !error && (
          <p style={{ fontSize: 13, color: "#9ca3af" }}>
            No documents uploaded yet. Upload a file using the button above.
          </p>
        )}

        {docs.length > 0 && (
          <section
            id="document-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: 20,
            }}
          >
            {docs.map((doc) => {
              const total = doc.total_chunks ?? 0;
              const risk = doc.risk_chunks ?? 0;
              const noRisk = doc.no_risk_chunks ?? 0;
              const score =
                doc.risk_score ?? (total ? risk / total : null);

              const extension = getFileExtension(doc.file_name);
              const isPdf = extension === "PDF";
              const previewUrl = doc.document_link || undefined;

              return (
                <article
                  key={doc.id}
                  onClick={() => handleOpenDetails(doc.id)}
                  style={{
                    cursor: "pointer",
                    borderRadius: 20,
                    border: "1px solid #1f2937",
                    padding: 16,
                    background:
                      "radial-gradient(circle at top, rgba(15,23,42,0.8), #020617 60%)",
                    boxShadow: "0 18px 40px rgba(0,0,0,0.65)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                    transition:
                      "transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s ease-out",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLDivElement).style.transform =
                      "translateY(-3px)";
                    (e.currentTarget as HTMLDivElement).style.boxShadow =
                      "0 25px 60px rgba(0,0,0,0.8)";
                    (e.currentTarget as HTMLDivElement).style.borderColor =
                      "#1d4ed8";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLDivElement).style.transform =
                      "translateY(0)";
                    (e.currentTarget as HTMLDivElement).style.boxShadow =
                      "0 18px 40px rgba(0,0,0,0.65)";
                    (e.currentTarget as HTMLDivElement).style.borderColor =
                      "#1f2937";
                  }}
                >
                  <div
                    style={{
                      borderRadius: 14,
                      background:
                        "linear-gradient(135deg, #0f172a, #020617)",
                      border: "1px solid #1f2937",
                      padding: 12,
                      minHeight: 120,
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "space-between",
                      gap: 8,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        fontSize: 11,
                        color: "#9ca3af",
                      }}
                    >
                      <span>Preview</span>
                      {extension && (
                        <span
                          style={{
                            padding: "2px 8px",
                            borderRadius: 999,
                            border: "1px solid #1f2937",
                          }}
                        >
                          {extension}
                        </span>
                      )}
                    </div>

                    {isPdf && previewUrl ? (
                      <div
                        style={{
                          marginTop: 6,
                          borderRadius: 10,
                          overflow: "hidden",
                          border: "1px solid #1f2937",
                          height: 110,
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <iframe
                          src={`${previewUrl}#toolbar=0&navpanes=0&scrollbar=0`}
                          style={{
                            width: "100%",
                            height: "100%",
                            border: "none",
                          }}
                          title={doc.file_name || "PDF preview"}
                        />
                      </div>
                    ) : (
                      <div
                        style={{
                          marginTop: 4,
                          padding: 8,
                          borderRadius: 10,
                          border: "1px dashed #1f2937",
                          display: "flex",
                          flexDirection: "column",
                          gap: 4,
                        }}
                      >
                        <div
                          style={{
                            height: 4,
                            borderRadius: 999,
                            background:
                              "linear-gradient(to right, #22c55e, transparent)",
                          }}
                        />
                        <div
                          style={{
                            height: 4,
                            borderRadius: 999,
                            background:
                              "linear-gradient(to right, #38bdf8, transparent)",
                          }}
                        />
                        <div
                          style={{
                            height: 4,
                            borderRadius: 999,
                            background:
                              "linear-gradient(to right, #fbbf24, transparent)",
                          }}
                        />
                      </div>
                    )}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <span
                      style={{
                        fontWeight: 500,
                        fontSize: 14,
                        maxWidth: "100%",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      title={doc.file_name ?? undefined}
                    >
                      {doc.file_name || "Untitled document"}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        color: "#9ca3af",
                        marginTop: 2,
                      }}
                    >
                      {new Date(doc.created_at).toLocaleString()}
                    </span>

                    {doc.document_link && (
                      <a
                        href={doc.document_link}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          fontSize: 11,
                          color: "#38bdf8",
                          marginTop: 4,
                        }}
                      >
                        View source
                      </a>
                    )}
                  </div>

                  <div
                    style={{
                      marginTop: "auto",
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 8,
                      fontSize: 11,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-start",
                      }}
                    >
                      <span style={{ color: "#6b7280" }}>Risk</span>
                      <span style={{ color: "#fbbf24", fontWeight: 600 }}>
                        {risk || "—"}
                      </span>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-start",
                      }}
                    >
                      <span style={{ color: "#6b7280" }}>No risk</span>
                      <span style={{ color: "#34d399", fontWeight: 600 }}>
                        {noRisk || "—"}
                      </span>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-end",
                        marginLeft: "auto",
                      }}
                    >
                      <span style={{ color: "#6b7280" }}>Risk score</span>
                      <span style={{ fontWeight: 600 }}>
                        {score !== null ? `${(score * 100).toFixed(1)}%` : "—"}
                      </span>
                    </div>
                  </div>
                </article>
              );
            })}
          </section>
        )}
      </div>
    </main>
  );
}

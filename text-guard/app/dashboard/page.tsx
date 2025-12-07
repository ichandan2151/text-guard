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
};

// Cloud Run URL (set in .env.local)
const PROCESSOR_URL = process.env.NEXT_PUBLIC_PROCESSOR_URL;

// ---- helper to call Cloud Run processor ----
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

  console.log("Calling processor:", url, {
    document_id: documentId,
    storage_path: storagePath,
    file_name: fileName,
  });

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
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

  const data = await res.json();
  console.log("Processor response:", data);
  return data;
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

  console.log("Processor URL (client):", PROCESSOR_URL);

  // ---------- auth check + load documents ----------
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

  // ---------- logout ----------
  const handleLogout = async () => {
    await supabase.auth.signOut();
    router.push("/login");
  };

  // ---------- upload handlers ----------
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

      // 1) Upload to Supabase Storage bucket "documents"
      const { data: storageData, error: storageError } = await supabase.storage
        .from("documents") // bucket name
        .upload(filePath, file);

      if (storageError || !storageData) {
        console.error(storageError);
        throw new Error("Failed to upload file to storage.");
      }

      console.log("Uploaded to storage:", storageData.path);

      // 2) Insert row into "documents" table and get the row back
      const { data: inserted, error: insertError } = await supabase
        .from("documents")
        .insert({
          file_name: file.name,
          storage_path: storageData.path,
          document_link: null,
          project_name: null,
          total_chunks: null,
          risk_chunks: null,
          no_risk_chunks: null,
          risk_score: null,
        })
        .select("*")
        .limit(1);

      if (insertError) {
        console.error(insertError);
        throw new Error("File stored, but failed to save metadata.");
      }

      const docRow = inserted?.[0] as DocumentRow | undefined;
      console.log("Inserted documents row:", docRow);

      // 3) Call Cloud Run processor to chunk + run BERT + update DB
      if (docRow) {
        await callProcessor({
          documentId: docRow.id,
          storagePath: storageData.path,
          fileName: file.name,
        });
      }

      // 4) Refresh table
      await fetchDocuments();

      // reset selection
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } finally {
      setUploading(false);
    }
  };

  // ---------- UI ----------
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
      <div style={{ width: "100%", maxWidth: 1000 }}>
        {/* Top bar: title + logout */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
            marginBottom: 16,
          }}
        >
          <div>
            <h1 style={{ fontSize: 24, margin: 0 }}>TextGuard Dashboard</h1>
            <p style={{ fontSize: 13, color: "#9ca3af", marginTop: 4 }}>
              View scanned documents and upload new files for risk analysis.
            </p>
          </div>

          <button
            onClick={handleLogout}
            style={{
              padding: "6px 14px",
              borderRadius: 999,
              border: "1px solid #374151",
              backgroundColor: "transparent",
              color: "#e5e7eb",
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            Log out
          </button>
        </div>

        {/* Upload controls */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: 16,
            marginBottom: 24,
            flexWrap: "wrap",
          }}
        >
          <div style={{ fontSize: 13, color: "#9ca3af" }}>
            Upload a new project document (PDF / DOCX) to run TextGuard on it.
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {/* Hidden input – opened by the button */}
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
                padding: "10px 18px",
                borderRadius: 999,
                border: "none",
                backgroundColor: "#22c55e",
                color: "#022c22",
                fontSize: 13,
                fontWeight: 500,
                cursor: uploading ? "default" : "pointer",
                opacity: uploading ? 0.7 : 1,
                minWidth: 140,
              }}
            >
              {uploading ? "Uploading…" : "Upload file"}
            </button>

            {selectedFile && !uploading && (
              <span style={{ fontSize: 12, color: "#e5e7eb" }}>
                Selected: {selectedFile.name}
              </span>
            )}

            {uploadError && (
              <span style={{ color: "#f97373", fontSize: 12 }}>
                {uploadError}
              </span>
            )}
          </div>
        </div>

        {/* Document log card */}
        <section
          style={{
            backgroundColor: "#020617",
            borderRadius: 20,
            border: "1px solid #1f2937",
            padding: 20,
            boxShadow: "0 20px 45px rgba(0,0,0,0.7)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 12,
            }}
          >
            <h2
              style={{
                fontSize: 13,
                letterSpacing: 1,
                textTransform: "uppercase",
                color: "#9ca3af",
              }}
            >
              Document log
            </h2>
            {loading && (
              <span style={{ fontSize: 12, color: "#9ca3af" }}>Loading…</span>
            )}
          </div>

          {error && (
            <p style={{ fontSize: 13, color: "#f97373", marginBottom: 8 }}>
              {error}
            </p>
          )}

          {!loading && docs.length === 0 && !error && (
            <p style={{ fontSize: 13, color: "#9ca3af" }}>
              No documents uploaded yet. Upload a file using the control above.
            </p>
          )}

          {docs.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 13,
                }}
              >
                <thead>
                  <tr
                    style={{
                      borderBottom: "1px solid #1f2937",
                      color: "#9ca3af",
                    }}
                  >
                    <th style={{ textAlign: "left", padding: "8px 8px" }}>
                      File
                    </th>
                    <th style={{ textAlign: "left", padding: "8px 8px" }}>
                      Uploaded
                    </th>
                    <th style={{ textAlign: "right", padding: "8px 8px" }}>
                      Chunks
                    </th>
                    <th style={{ textAlign: "right", padding: "8px 8px" }}>
                      Risk
                    </th>
                    <th style={{ textAlign: "right", padding: "8px 8px" }}>
                      No risk
                    </th>
                    <th style={{ textAlign: "right", padding: "8px 8px" }}>
                      Risk score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc) => {
                    const total = doc.total_chunks ?? 0;
                    const risk = doc.risk_chunks ?? 0;
                    const noRisk = doc.no_risk_chunks ?? 0;
                    const score =
                      doc.risk_score ?? (total ? risk / total : null);

                    return (
                      <tr
                        key={doc.id}
                        style={{ borderBottom: "1px solid #111827" }}
                      >
                        <td style={{ padding: "8px 8px" }}>
                          <div
                            style={{
                              display: "flex",
                              flexDirection: "column",
                            }}
                          >
                            <span
                              style={{
                                fontWeight: 500,
                                maxWidth: 260,
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                              }}
                            >
                              {doc.file_name}
                            </span>
                            {doc.document_link && (
                              <a
                                href={doc.document_link}
                                target="_blank"
                                rel="noreferrer"
                                style={{
                                  fontSize: 11,
                                  color: "#38bdf8",
                                  marginTop: 2,
                                }}
                              >
                                View source
                              </a>
                            )}
                          </div>
                        </td>
                        <td style={{ padding: "8px 8px", color: "#e5e7eb" }}>
                          {new Date(doc.created_at).toLocaleString()}
                        </td>
                        <td style={{ padding: "8px 8px", textAlign: "right" }}>
                          {total || "—"}
                        </td>
                        <td
                          style={{
                            padding: "8px 8px",
                            textAlign: "right",
                            color: "#fbbf24",
                          }}
                        >
                          {risk || "—"}
                        </td>
                        <td
                          style={{
                            padding: "8px 8px",
                            textAlign: "right",
                            color: "#34d399",
                          }}
                        >
                          {noRisk || "—"}
                        </td>
                        <td style={{ padding: "8px 8px", textAlign: "right" }}>
                          {score !== null
                            ? `${(score * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

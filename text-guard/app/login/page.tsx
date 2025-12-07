// app/login/page.tsx
"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "../lib/supabaseClient";

export default function AuthPage() {
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setMessage(null);
    setLoading(true);

    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({ email, password });

        if (error) {
          setMessage(error.message);
        } else {
          setMessage("Check your email to confirm your account.");
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) {
          setMessage(error.message);
        } else {
          setMessage("Logged in successfully!");
          router.push("/dashboard");
        }
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    setLoading(true);
    try {
      await supabase.auth.signOut();
      setMessage("Logged out.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        style={{
          maxWidth: 400,
          width: "100%",
          padding: 24,
          borderRadius: 16,
          border: "1px solid #1f2937",
          background:
            "radial-gradient(circle at top, rgba(55,65,81,0.6), #020617 55%)",
          boxShadow: "0 24px 60px rgba(0,0,0,0.7)",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <h1 style={{ fontSize: 22, margin: 0, textAlign: "center" }}>
          {mode === "login" ? "Log in" : "Sign up"}
        </h1>

        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", flexDirection: "column", gap: 12 }}
        >
          <label style={{ fontSize: 14 }}>
            Email
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{
                width: "100%",
                padding: 8,
                marginTop: 4,
                borderRadius: 8,
                border: "1px solid #374151",
                backgroundColor: "#020617",
                color: "#f9fafb",
              }}
            />
          </label>

          <label style={{ fontSize: 14 }}>
            Password
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{
                width: "100%",
                padding: 8,
                marginTop: 4,
                borderRadius: 8,
                border: "1px solid #374151",
                backgroundColor: "#020617",
                color: "#f9fafb",
              }}
            />
          </label>

          <button
            type="submit"
            disabled={loading}
            style={{
              padding: 10,
              marginTop: 8,
              cursor: loading ? "default" : "pointer",
              borderRadius: 999,
              border: "none",
              background:
                "linear-gradient(to right, #22c55e, #22c55e, #16a34a)",
              color: "#020617",
              fontWeight: 600,
              fontSize: 14,
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading
              ? "Please wait..."
              : mode === "login"
              ? "Log in"
              : "Create account"}
          </button>
        </form>

        <button
          type="button"
          onClick={() => setMode(mode === "login" ? "signup" : "login")}
          style={{
            marginTop: 8,
            fontSize: 13,
            background: "none",
            border: "none",
            color: "#9ca3af",
            cursor: "pointer",
          }}
        >
          {mode === "login"
            ? "Don't have an account? Sign up"
            : "Already have an account? Log in"}
        </button>

        <button
          type="button"
          onClick={handleLogout}
          disabled={loading}
          style={{
            marginTop: 4,
            fontSize: 13,
            background: "none",
            border: "none",
            color: "#6b7280",
            cursor: loading ? "default" : "pointer",
          }}
        >
          Log out
        </button>

        {message && (
          <p style={{ marginTop: 8, fontSize: 13, color: "#e5e7eb" }}>
            {message}
          </p>
        )}
      </div>
    </main>
  );
}

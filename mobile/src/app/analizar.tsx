import { router, useLocalSearchParams } from "expo-router";
import { useEffect, useState } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { createJob, getJob, statusIndex, STATUS_STEPS, type JobStatus } from "@/lib/api";
import { saveRecipe } from "@/lib/storage";
import { useTheme } from "@/lib/theme";

const POLL_MS = 2000;
const TIMEOUT_MS = 15 * 60 * 1000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function Analizar() {
  const theme = useTheme();
  const { text } = useLocalSearchParams<{ text?: string }>();
  const [status, setStatus] = useState<JobStatus>("queued");
  const [error, setError] = useState<string | null>(null);
  const [notRecipe, setNotRecipe] = useState<string | null>(null);
  // El clasificador propio rechazó el video antes de analizarlo: se puede insistir.
  const [rejected, setRejected] = useState(false);
  const [force, setForce] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        if (!text) throw new Error("No llegó ningún enlace.");
        let job = await createJob(text, "español", force);
        const started = Date.now();
        while (job.status !== "done" && job.status !== "error") {
          if (cancelled) return;
          if (Date.now() - started > TIMEOUT_MS) throw new Error("El análisis está tardando demasiado.");
          setStatus(job.status);
          await sleep(POLL_MS);
          job = await getJob(job.id);
        }
        if (cancelled) return;
        if (job.status === "error" && job.error_code === "not_recipe") {
          setRejected(true);
          setNotRecipe(job.error);
          return;
        }
        if (job.status === "error" || !job.recipe) {
          throw new Error(job.error ?? "No se pudo sacar la receta.");
        }
        if (!job.recipe.is_recipe) {
          setNotRecipe(job.recipe.description);
          return;
        }
        await saveRecipe({ id: job.id, savedAt: Date.now(), recipe: job.recipe, source: job.source });
        router.replace({ pathname: "/receta/[id]", params: { id: job.id } });
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [text, attempt, force]);

  if (error || notRecipe) {
    return (
      <View style={styles.center}>
        <Text style={[styles.heading, { color: theme.text }]}>
          {error ? "No pude sacar la receta" : "Este video no parece una receta"}
        </Text>
        <Text style={[styles.body, { color: theme.muted }]}>{error ?? notRecipe}</Text>
        {error && (
          <Pressable
            onPress={() => {
              setError(null);
              setStatus("queued");
              setAttempt((n) => n + 1);
            }}
            style={[styles.button, { backgroundColor: theme.accent }]}
            accessibilityRole="button"
          >
            <Text style={[styles.buttonText, { color: theme.onAccent }]}>Intentar de nuevo</Text>
          </Pressable>
        )}
        {rejected && (
          <Pressable
            onPress={() => {
              setRejected(false);
              setNotRecipe(null);
              setStatus("queued");
              setForce(true);
            }}
            style={[styles.button, { backgroundColor: theme.accent }]}
            accessibilityRole="button"
          >
            <Text style={[styles.buttonText, { color: theme.onAccent }]}>Analizar de todos modos</Text>
          </Pressable>
        )}
        <Pressable onPress={() => router.back()} accessibilityRole="button">
          <Text style={[styles.link, { color: theme.accent }]}>Volver</Text>
        </Pressable>
      </View>
    );
  }

  const current = statusIndex(status);

  return (
    <View style={styles.center}>
      <Text style={[styles.heading, { color: theme.text }]}>Preparando tu receta…</Text>
      <Text style={[styles.body, { color: theme.muted }]}>Suele tardar entre 30 segundos y 2 minutos.</Text>
      <View style={styles.steps}>
        {STATUS_STEPS.map((step) => {
          const idx = statusIndex(step.status);
          const done = current > idx;
          const active = current === idx;
          return (
            <View key={step.status} style={styles.stepRow}>
              <View style={styles.stepIcon}>
                {active ? (
                  <ActivityIndicator color={theme.accent} />
                ) : (
                  <Text style={{ color: done ? theme.accent : theme.border, fontSize: 18 }}>{done ? "✓" : "○"}</Text>
                )}
              </View>
              <Text style={[styles.stepLabel, { color: done || active ? theme.text : theme.muted }]}>{step.label}</Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: "center", padding: 24, gap: 14, width: "100%", maxWidth: 520, alignSelf: "center" },
  heading: { fontSize: 24, fontWeight: "700" },
  body: { fontSize: 16, lineHeight: 22 },
  steps: { marginTop: 12, gap: 14 },
  stepRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  stepIcon: { width: 24, alignItems: "center" },
  stepLabel: { fontSize: 17 },
  button: { borderRadius: 14, paddingVertical: 14, alignItems: "center", marginTop: 8 },
  buttonText: { fontSize: 17, fontWeight: "700" },
  link: { fontSize: 16, textAlign: "center", paddingVertical: 8 },
});

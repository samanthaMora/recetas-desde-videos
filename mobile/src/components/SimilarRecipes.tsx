import { router } from "expo-router";
import { useEffect, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { similarRecipes, type SimilarHit } from "@/lib/api";
import { useTheme } from "@/lib/theme";

// "Recetas parecidas": vecinos más cercanos en el espacio de embeddings del servidor.
export function SimilarRecipes({ id }: { id: string }) {
  const theme = useTheme();
  const [hits, setHits] = useState<SimilarHit[]>([]);

  useEffect(() => {
    let cancelled = false;
    similarRecipes(id)
      .then((h) => !cancelled && setHits(h.filter((x) => x.job.recipe?.is_recipe)))
      .catch(() => undefined); // sin conexión: simplemente no se muestra la sección
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (!hits.length) return null;

  return (
    <View style={styles.section}>
      <Text style={[styles.title, { color: theme.text }]}>Recetas parecidas</Text>
      {hits.map(({ job }) => (
        <Pressable
          key={job.id}
          onPress={() => router.push({ pathname: "/receta/[id]", params: { id: job.id } })}
          style={({ pressed }) => [
            styles.card,
            { backgroundColor: theme.card, borderColor: theme.border, opacity: pressed ? 0.8 : 1 },
          ]}
          accessibilityRole="button"
        >
          <Text style={[styles.cardTitle, { color: theme.text }]} numberOfLines={2}>
            {job.recipe?.title}
          </Text>
          {!!job.source?.uploader && (
            <Text style={[styles.cardMeta, { color: theme.muted }]} numberOfLines={1}>
              {job.source.uploader}
            </Text>
          )}
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  section: { marginTop: 32, gap: 10 },
  title: { fontSize: 21, fontWeight: "700" },
  card: { borderWidth: 1, borderRadius: 14, padding: 14, gap: 4 },
  cardTitle: { fontSize: 16, fontWeight: "600" },
  cardMeta: { fontSize: 14 },
});

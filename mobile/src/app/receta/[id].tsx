import { router, Stack, useFocusEffect, useLocalSearchParams } from "expo-router";
import { useCallback, useState } from "react";
import { ActivityIndicator, Alert, Platform, Pressable, StyleSheet, Text, View } from "react-native";

import { RecipeView } from "@/components/RecipeView";
import { SimilarRecipes } from "@/components/SimilarRecipes";
import { getJob } from "@/lib/api";
import { deleteRecipe, getRecipe, saveRecipe, type SavedRecipe } from "@/lib/storage";
import { useTheme } from "@/lib/theme";

function confirmDelete(onConfirm: () => void) {
  if (Platform.OS === "web") {
    if (window.confirm("¿Eliminar esta receta?")) onConfirm();
    return;
  }
  Alert.alert("¿Eliminar esta receta?", undefined, [
    { text: "Cancelar", style: "cancel" },
    { text: "Eliminar", style: "destructive", onPress: onConfirm },
  ]);
}

// Receta guardada en el dispositivo, o (si viene de "Recetas parecidas") pedida al servidor.
type Loaded = { item: SavedRecipe; saved: boolean } | null;

async function load(id: string): Promise<Loaded> {
  const local = await getRecipe(id);
  if (local) return { item: local, saved: true };
  try {
    const job = await getJob(id);
    if (job.status === "done" && job.recipe) {
      return { item: { id: job.id, savedAt: Date.now(), recipe: job.recipe, source: job.source }, saved: false };
    }
  } catch {
    // sin conexión o no existe
  }
  return null;
}

export default function RecetaScreen() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [loaded, setLoaded] = useState<Loaded | undefined>(undefined);

  // Se recarga al volver de la pantalla de edición.
  useFocusEffect(
    useCallback(() => {
      load(id).then(setLoaded);
    }, [id]),
  );

  if (loaded === undefined) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={theme.accent} />
      </View>
    );
  }

  if (loaded === null) {
    return (
      <View style={styles.center}>
        <Text style={{ color: theme.muted, fontSize: 16 }}>No encontré esta receta.</Text>
      </View>
    );
  }

  const { item, saved } = loaded;

  const headerRight = saved
    ? () => (
        <View style={styles.headerActions}>
          <Pressable
            onPress={() => router.push({ pathname: "/editar/[id]", params: { id: item.id } })}
            hitSlop={10}
            accessibilityRole="button"
          >
            <Text style={{ color: theme.accent, fontSize: 16 }}>Editar</Text>
          </Pressable>
          <Pressable
            onPress={() =>
              confirmDelete(async () => {
                await deleteRecipe(item.id);
                router.back();
              })
            }
            hitSlop={10}
            accessibilityRole="button"
          >
            <Text style={{ color: theme.danger, fontSize: 16 }}>Eliminar</Text>
          </Pressable>
        </View>
      )
    : () => (
        <View style={styles.headerActions}>
          <Pressable
            onPress={async () => {
              await saveRecipe({ ...item, savedAt: Date.now() });
              setLoaded({ item, saved: true });
            }}
            hitSlop={10}
            accessibilityRole="button"
          >
            <Text style={{ color: theme.accent, fontSize: 16, fontWeight: "600" }}>Guardar</Text>
          </Pressable>
        </View>
      );

  return (
    <>
      <Stack.Screen options={{ title: "", headerRight }} />
      <RecipeView recipe={item.recipe} source={item.source} footer={<SimilarRecipes id={item.id} />} />
    </>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24 },
  headerActions: { flexDirection: "row", gap: 20, paddingRight: Platform.OS === "web" ? 16 : 0 },
});

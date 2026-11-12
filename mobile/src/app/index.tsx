import { router, useFocusEffect, useLocalSearchParams } from "expo-router";
import { useShareIntentContext } from "expo-share-intent";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { searchRecipes } from "@/lib/api";
import { loadRecipes, type SavedRecipe } from "@/lib/storage";
import { useTheme } from "@/lib/theme";

function analyze(text: string) {
  router.push({ pathname: "/analizar", params: { text } });
}

const fold = (s: string) =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "");

// Filtro por texto: respaldo inmediato mientras responde (o si falla) la búsqueda semántica.
function textFilter(list: SavedRecipe[], query: string): SavedRecipe[] {
  const words = fold(query).split(/\s+/).filter((w) => w.length >= 3);
  if (!words.length) return list;
  return list.filter((r) => {
    const text = fold([r.recipe.title, r.recipe.description, ...r.recipe.ingredients.map((i) => i.name)].join(" "));
    return words.some((w) => text.includes(w));
  });
}

export default function Home() {
  const theme = useTheme();
  const [input, setInput] = useState("");
  const [saved, setSaved] = useState<SavedRecipe[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ query: string; ids: string[] } | null>(null);
  const { hasShareIntent, shareIntent, resetShareIntent } = useShareIntentContext();
  // En la web se puede abrir la app con ?url=<enlace del video>
  const params = useLocalSearchParams<{ url?: string; text?: string }>();

  // Video compartido desde Instagram, TikTok, YouTube, etc.
  useEffect(() => {
    if (!hasShareIntent) return;
    const text = shareIntent.webUrl || shareIntent.text;
    resetShareIntent();
    if (text) analyze(text);
  }, [hasShareIntent, shareIntent, resetShareIntent]);

  useEffect(() => {
    const text = params.url || params.text;
    if (text) {
      router.setParams({ url: undefined, text: undefined });
      analyze(text);
    }
  }, [params.url, params.text]);

  useFocusEffect(
    useCallback(() => {
      loadRecipes().then(setSaved);
    }, []),
  );

  // Búsqueda semántica en el servidor, 300 ms después de dejar de escribir.
  useEffect(() => {
    const q = query.trim();
    if (!q || !saved.length) return;
    const timer = setTimeout(() => {
      searchRecipes(q, saved.map((r) => r.id))
        .then((hits) => setResults({ query: q, ids: hits.map((h) => h.id) }))
        .catch(() => setResults({ query: q, ids: textFilter(saved, q).map((r) => r.id) }));
    }, 300);
    return () => clearTimeout(timer);
  }, [query, saved]);

  const visible = useMemo(() => {
    const q = query.trim();
    if (!q) return saved;
    if (results?.query !== q) return textFilter(saved, q);
    const byId = new Map(saved.map((r) => [r.id, r]));
    // Las recetas que el servidor aún no tiene indexadas se buscan por texto.
    const semantic = results.ids.map((id) => byId.get(id)).filter((r): r is SavedRecipe => !!r);
    const extra = textFilter(saved, q).filter((r) => !results.ids.includes(r.id));
    return [...semantic, ...extra];
  }, [query, results, saved]);

  const submit = () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    analyze(text);
  };

  const header = (
    <View style={styles.header}>
      <Text style={[styles.title, { color: theme.text }]}>Recetas</Text>
      <Text style={[styles.subtitle, { color: theme.muted }]}>
        Pega el enlace de un video de cocina y te escribo la receta.
      </Text>

      <View style={[styles.inputRow, { backgroundColor: theme.card, borderColor: theme.border }]}>
        <TextInput
          value={input}
          onChangeText={setInput}
          onSubmitEditing={submit}
          placeholder="https://www.instagram.com/reel/…"
          placeholderTextColor={theme.muted}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          returnKeyType="go"
          style={[styles.input, { color: theme.text }]}
          accessibilityLabel="Enlace del video"
        />
      </View>
      <Pressable
        onPress={submit}
        disabled={!input.trim()}
        style={({ pressed }) => [
          styles.button,
          { backgroundColor: theme.accent, opacity: !input.trim() ? 0.5 : pressed ? 0.85 : 1 },
        ]}
        accessibilityRole="button"
      >
        <Text style={[styles.buttonText, { color: theme.onAccent }]}>Sacar la receta</Text>
      </Pressable>

      {Platform.OS !== "web" && (
        <Text style={[styles.hint, { color: theme.muted }]}>
          También puedes tocar «Compartir» en Instagram, TikTok, YouTube o Facebook y elegir Recetas.
        </Text>
      )}

      <Text style={[styles.sectionTitle, { color: theme.text }]}>Mis recetas</Text>
      {saved.length > 0 && (
        <View style={[styles.inputRow, styles.searchRow, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <TextInput
            value={query}
            onChangeText={setQuery}
            placeholder="Buscar: «algo con pollo», «postre sin horno»…"
            placeholderTextColor={theme.muted}
            autoCorrect={false}
            returnKeyType="search"
            style={[styles.input, styles.searchInput, { color: theme.text }]}
            accessibilityLabel="Buscar en mis recetas"
          />
        </View>
      )}
    </View>
  );

  return (
    <SafeAreaView style={[styles.flex, { backgroundColor: theme.bg }]} edges={["top", "left", "right"]}>
      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <FlatList
          data={visible}
          keyExtractor={(item) => item.id}
          ListHeaderComponent={header}
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          ListEmptyComponent={
            <Text style={[styles.empty, { color: theme.muted }]}>
              {query.trim()
                ? `No encontré recetas para «${query.trim()}».`
                : "Aún no tienes recetas. La primera que saques aparecerá aquí."}
            </Text>
          }
          renderItem={({ item }) => (
            <Pressable
              onPress={() => router.push({ pathname: "/receta/[id]", params: { id: item.id } })}
              style={({ pressed }) => [
                styles.card,
                { backgroundColor: theme.card, borderColor: theme.border, opacity: pressed ? 0.8 : 1 },
              ]}
              accessibilityRole="button"
            >
              <Text style={[styles.cardTitle, { color: theme.text }]} numberOfLines={2}>
                {item.recipe.title}
              </Text>
              <Text style={[styles.cardMeta, { color: theme.muted }]} numberOfLines={1}>
                {[item.source?.uploader, `${item.recipe.ingredients.length} ingredientes`]
                  .filter(Boolean)
                  .join(" · ")}
              </Text>
            </Pressable>
          )}
        />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { padding: 16, paddingBottom: 48, width: "100%", maxWidth: 640, alignSelf: "center" },
  header: { gap: 12, marginBottom: 8 },
  title: { fontSize: 34, fontWeight: "800", marginTop: 12 },
  subtitle: { fontSize: 16, lineHeight: 22 },
  inputRow: { borderWidth: 1, borderRadius: 14, marginTop: 8 },
  input: { fontSize: 16, paddingHorizontal: 14, paddingVertical: 14 },
  button: { borderRadius: 14, paddingVertical: 15, alignItems: "center" },
  buttonText: { fontSize: 17, fontWeight: "700" },
  hint: { fontSize: 14, lineHeight: 20 },
  sectionTitle: { fontSize: 20, fontWeight: "700", marginTop: 20 },
  searchRow: { marginTop: 0, marginBottom: 4 },
  searchInput: { paddingVertical: 11, fontSize: 15 },
  empty: { fontSize: 15, marginTop: 4 },
  card: { borderWidth: 1, borderRadius: 14, padding: 14, marginBottom: 10, gap: 4 },
  cardTitle: { fontSize: 17, fontWeight: "600" },
  cardMeta: { fontSize: 14 },
});

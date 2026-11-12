import { router, useLocalSearchParams } from "expo-router";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { sendCorrection, type Ingredient, type Recipe, type Step } from "@/lib/api";
import { getRecipe, saveRecipe, type SavedRecipe } from "@/lib/storage";
import { useTheme, type Theme } from "@/lib/theme";

const emptyIngredient = (): Ingredient => ({ name: "", quantity: null, unit: null, notes: null, estimated: false });
const emptyStep = (n: number): Step => ({ number: n, instruction: "", duration_minutes: null, timestamp: null });
const orNull = (s: string) => (s.trim() ? s : null);

function Field({
  theme,
  style,
  ...props
}: React.ComponentProps<typeof TextInput> & { theme: Theme }) {
  return (
    <TextInput
      placeholderTextColor={theme.muted}
      style={[styles.input, { color: theme.text, backgroundColor: theme.card, borderColor: theme.border }, style]}
      {...props}
    />
  );
}

export default function EditarScreen() {
  const theme = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [item, setItem] = useState<SavedRecipe | null>(null);
  const [draft, setDraft] = useState<Recipe | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getRecipe(id).then((saved) => {
      setItem(saved);
      setDraft(saved ? structuredClone(saved.recipe) : null);
    });
  }, [id]);

  if (!item || !draft) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={theme.accent} />
      </View>
    );
  }

  const update = (changes: Partial<Recipe>) => setDraft({ ...draft, ...changes });

  const updateIngredient = (idx: number, changes: Partial<Ingredient>) => {
    const ingredients = draft.ingredients.map((ing, i) => {
      if (i !== idx) return ing;
      const next = { ...ing, ...changes };
      // Si la persona escribe la cantidad, ya no es una estimación.
      if ("quantity" in changes || "unit" in changes) next.estimated = false;
      return next;
    });
    update({ ingredients });
  };

  const updateStep = (idx: number, instruction: string) =>
    update({ steps: draft.steps.map((s, i) => (i === idx ? { ...s, instruction } : s)) });

  const save = async () => {
    setSaving(true);
    const recipe: Recipe = {
      ...draft,
      title: draft.title.trim() || item.recipe.title,
      ingredients: draft.ingredients.filter((i) => i.name.trim()),
      steps: draft.steps.filter((s) => s.instruction.trim()).map((s, i) => ({ ...s, number: i + 1 })),
    };
    await saveRecipe({ ...item, recipe });
    // Si no hay conexión, la corrección se guarda igual en el teléfono; solo se pierde como dato de entrenamiento.
    await sendCorrection(item.id, recipe).catch(() => undefined);
    router.back();
  };

  return (
    <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={[styles.label, { color: theme.muted }]}>Título</Text>
        <Field theme={theme} value={draft.title} onChangeText={(title) => update({ title })} />

        <Text style={[styles.label, { color: theme.muted }]}>Porciones</Text>
        <Field
          theme={theme}
          value={draft.servings ?? ""}
          onChangeText={(s) => update({ servings: orNull(s) })}
          placeholder="p. ej. 4"
        />

        <Text style={[styles.section, { color: theme.text }]}>Ingredientes</Text>
        {draft.ingredients.map((ing, idx) => (
          <View key={idx} style={styles.row}>
            <Field
              theme={theme}
              style={styles.qty}
              value={ing.quantity ?? ""}
              onChangeText={(q) => updateIngredient(idx, { quantity: orNull(q) })}
              placeholder="Cant."
            />
            <Field
              theme={theme}
              style={styles.unit}
              value={ing.unit ?? ""}
              onChangeText={(u) => updateIngredient(idx, { unit: orNull(u) })}
              placeholder="Unidad"
            />
            <Field
              theme={theme}
              style={styles.flex}
              value={ing.name}
              onChangeText={(name) => updateIngredient(idx, { name })}
              placeholder="Ingrediente"
            />
            <Pressable
              onPress={() => update({ ingredients: draft.ingredients.filter((_, i) => i !== idx) })}
              hitSlop={8}
              accessibilityLabel="Quitar ingrediente"
            >
              <Text style={[styles.remove, { color: theme.muted }]}>✕</Text>
            </Pressable>
          </View>
        ))}
        <Pressable onPress={() => update({ ingredients: [...draft.ingredients, emptyIngredient()] })}>
          <Text style={[styles.add, { color: theme.accent }]}>+ Añadir ingrediente</Text>
        </Pressable>

        <Text style={[styles.section, { color: theme.text }]}>Pasos</Text>
        {draft.steps.map((step, idx) => (
          <View key={idx} style={styles.row}>
            <Text style={[styles.stepNumber, { color: theme.muted }]}>{idx + 1}.</Text>
            <Field
              theme={theme}
              style={[styles.flex, styles.multiline]}
              value={step.instruction}
              onChangeText={(t) => updateStep(idx, t)}
              multiline
              placeholder="Describe el paso"
            />
            <Pressable
              onPress={() => update({ steps: draft.steps.filter((_, i) => i !== idx) })}
              hitSlop={8}
              accessibilityLabel="Quitar paso"
            >
              <Text style={[styles.remove, { color: theme.muted }]}>✕</Text>
            </Pressable>
          </View>
        ))}
        <Pressable onPress={() => update({ steps: [...draft.steps, emptyStep(draft.steps.length + 1)] })}>
          <Text style={[styles.add, { color: theme.accent }]}>+ Añadir paso</Text>
        </Pressable>

        <Pressable
          onPress={save}
          disabled={saving}
          style={[styles.button, { backgroundColor: theme.accent, opacity: saving ? 0.6 : 1 }]}
          accessibilityRole="button"
        >
          <Text style={[styles.buttonText, { color: theme.onAccent }]}>{saving ? "Guardando…" : "Guardar cambios"}</Text>
        </Pressable>
        <Text style={[styles.hint, { color: theme.muted }]}>
          Tus correcciones ayudan a que la app aprenda a sacar mejores recetas.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  content: { padding: 16, paddingBottom: 64, gap: 8, width: "100%", maxWidth: 680, alignSelf: "center" },
  label: { fontSize: 13, marginTop: 8 },
  section: { fontSize: 20, fontWeight: "700", marginTop: 20 },
  input: { borderWidth: 1, borderRadius: 10, paddingHorizontal: 10, paddingVertical: 10, fontSize: 16 },
  multiline: { minHeight: 64, textAlignVertical: "top" },
  row: { flexDirection: "row", gap: 6, alignItems: "flex-start" },
  qty: { width: 64 },
  unit: { width: 84 },
  remove: { fontSize: 18, paddingHorizontal: 6, paddingTop: 10 },
  stepNumber: { fontSize: 16, paddingTop: 11, width: 24 },
  add: { fontSize: 16, paddingVertical: 8 },
  button: { borderRadius: 14, paddingVertical: 14, alignItems: "center", marginTop: 24 },
  buttonText: { fontSize: 17, fontWeight: "700" },
  hint: { fontSize: 13, textAlign: "center", marginTop: 8 },
});

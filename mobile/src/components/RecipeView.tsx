import { useState } from "react";
import { Linking, Pressable, ScrollView, Share, StyleSheet, Text, View } from "react-native";

import { recipeAsText, type Ingredient, type Recipe, type SourceInfo } from "@/lib/api";
import { useTheme, type Theme } from "@/lib/theme";

function minutes(n: number | null): string | null {
  if (n == null) return null;
  if (n < 60) return `${Math.round(n)} min`;
  const h = Math.floor(n / 60);
  const m = Math.round(n % 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

function Chip({ label, value, theme }: { label: string; value: string; theme: Theme }) {
  return (
    <View style={[styles.chip, { backgroundColor: theme.accentSoft }]}>
      <Text style={[styles.chipLabel, { color: theme.muted }]}>{label}</Text>
      <Text style={[styles.chipValue, { color: theme.text }]}>{value}</Text>
    </View>
  );
}

function Section({ title, theme, children }: { title: string; theme: Theme; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={[styles.sectionTitle, { color: theme.text }]}>{title}</Text>
      {children}
    </View>
  );
}

function ingredientText(i: Ingredient): string {
  const qty = [i.quantity, i.unit].filter(Boolean).join(" ");
  return `${qty ? (i.estimated ? "≈ " : "") + qty + " " : ""}${i.name}`;
}

export function RecipeView({
  recipe,
  source,
  footer,
}: {
  recipe: Recipe;
  source: SourceInfo | null;
  footer?: React.ReactNode;
}) {
  const theme = useTheme();
  const [checked, setChecked] = useState<Set<number>>(new Set());

  const toggle = (idx: number) =>
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });

  const chips = [
    recipe.servings && { label: "Porciones", value: recipe.servings },
    minutes(recipe.prep_time_minutes) && { label: "Preparación", value: minutes(recipe.prep_time_minutes)! },
    minutes(recipe.cook_time_minutes) && { label: "Cocción", value: minutes(recipe.cook_time_minutes)! },
    minutes(recipe.total_time_minutes) && { label: "Total", value: minutes(recipe.total_time_minutes)! },
    recipe.difficulty && { label: "Dificultad", value: recipe.difficulty },
  ].filter(Boolean) as { label: string; value: string }[];

  const hasEstimates = recipe.ingredients.some((i) => i.estimated && (i.quantity || i.unit));

  return (
    <ScrollView contentContainerStyle={styles.content}>
      <Text style={[styles.title, { color: theme.text }]}>{recipe.title}</Text>
      {!!recipe.description && <Text style={[styles.description, { color: theme.muted }]}>{recipe.description}</Text>}

      {chips.length > 0 && (
        <View style={styles.chips}>
          {chips.map((c) => (
            <Chip key={c.label} {...c} theme={theme} />
          ))}
        </View>
      )}

      <Section title="Ingredientes" theme={theme}>
        {recipe.ingredients.map((ing, idx) => {
          const done = checked.has(idx);
          return (
            <Pressable
              key={idx}
              onPress={() => toggle(idx)}
              style={styles.ingredientRow}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: done }}
            >
              <View
                style={[
                  styles.checkbox,
                  { borderColor: done ? theme.accent : theme.border, backgroundColor: done ? theme.accent : "transparent" },
                ]}
              >
                {done && <Text style={{ color: theme.onAccent, fontSize: 13, fontWeight: "800" }}>✓</Text>}
              </View>
              <View style={styles.flex}>
                <Text
                  style={[
                    styles.ingredient,
                    { color: done ? theme.muted : theme.text, textDecorationLine: done ? "line-through" : "none" },
                  ]}
                >
                  {ingredientText(ing)}
                </Text>
                {!!ing.notes && <Text style={[styles.note, { color: theme.muted }]}>{ing.notes}</Text>}
              </View>
            </Pressable>
          );
        })}
        {hasEstimates && (
          <Text style={[styles.note, { color: theme.muted }]}>≈ cantidad estimada viendo el video; no se dijo exacta.</Text>
        )}
      </Section>

      <Section title="Preparación" theme={theme}>
        {recipe.steps.map((s) => (
          <View key={s.number} style={styles.stepRow}>
            <View style={[styles.stepNumber, { backgroundColor: theme.accent }]}>
              <Text style={{ color: theme.onAccent, fontWeight: "700" }}>{s.number}</Text>
            </View>
            <View style={styles.flex}>
              <Text style={[styles.stepText, { color: theme.text }]}>{s.instruction}</Text>
              {(s.duration_minutes != null || s.timestamp) && (
                <Text style={[styles.note, { color: theme.muted }]}>
                  {[minutes(s.duration_minutes), s.timestamp && `en el video: ${s.timestamp}`].filter(Boolean).join(" · ")}
                </Text>
              )}
            </View>
          </View>
        ))}
      </Section>

      {recipe.equipment.length > 0 && (
        <Section title="Utensilios" theme={theme}>
          <Text style={[styles.body, { color: theme.text }]}>{recipe.equipment.join(", ")}</Text>
        </Section>
      )}

      {recipe.tips.length > 0 && (
        <Section title="Consejos" theme={theme}>
          {recipe.tips.map((t, i) => (
            <Text key={i} style={[styles.body, { color: theme.text }]}>• {t}</Text>
          ))}
        </Section>
      )}

      {recipe.uncertainties.length > 0 && (
        <Section title="Lo que no quedó claro en el video" theme={theme}>
          {recipe.uncertainties.map((t, i) => (
            <Text key={i} style={[styles.body, { color: theme.muted }]}>• {t}</Text>
          ))}
        </Section>
      )}

      <View style={styles.actions}>
        <Pressable
          onPress={() =>
            // Cancelar la ventana de compartir rechaza la promesa: no es un error.
            Share.share({ message: recipeAsText(recipe, source), title: recipe.title }).catch(() => undefined)
          }
          style={[styles.button, { backgroundColor: theme.accent }]}
          accessibilityRole="button"
        >
          <Text style={[styles.buttonText, { color: theme.onAccent }]}>Compartir receta</Text>
        </Pressable>
        {source?.url && (
          <Pressable onPress={() => Linking.openURL(source.url)} accessibilityRole="link">
            <Text style={[styles.link, { color: theme.accent }]}>
              Ver el video original{source.uploader ? ` de ${source.uploader}` : ""}
            </Text>
          </Pressable>
        )}
      </View>
      {footer}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { padding: 16, paddingBottom: 48, gap: 12, width: "100%", maxWidth: 680, alignSelf: "center" },
  title: { fontSize: 28, fontWeight: "800", lineHeight: 34 },
  description: { fontSize: 16, lineHeight: 23 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  chip: { borderRadius: 12, paddingHorizontal: 12, paddingVertical: 8 },
  chipLabel: { fontSize: 12 },
  chipValue: { fontSize: 15, fontWeight: "600" },
  section: { marginTop: 16, gap: 10 },
  sectionTitle: { fontSize: 21, fontWeight: "700" },
  ingredientRow: { flexDirection: "row", gap: 12, alignItems: "flex-start", paddingVertical: 2 },
  checkbox: { width: 22, height: 22, borderRadius: 6, borderWidth: 2, alignItems: "center", justifyContent: "center", marginTop: 1 },
  ingredient: { fontSize: 16, lineHeight: 22 },
  note: { fontSize: 13, lineHeight: 18, marginTop: 2 },
  stepRow: { flexDirection: "row", gap: 12, alignItems: "flex-start" },
  stepNumber: { width: 28, height: 28, borderRadius: 14, alignItems: "center", justifyContent: "center" },
  stepText: { fontSize: 16, lineHeight: 23 },
  body: { fontSize: 16, lineHeight: 23 },
  actions: { marginTop: 24, gap: 12 },
  button: { borderRadius: 14, paddingVertical: 14, alignItems: "center" },
  buttonText: { fontSize: 17, fontWeight: "700" },
  link: { fontSize: 16, textAlign: "center", paddingVertical: 8 },
});

// Recetas guardadas en el dispositivo (en la web, en el navegador).
import AsyncStorage from "@react-native-async-storage/async-storage";

import type { Recipe, SourceInfo } from "./api";

const KEY = "recetas:guardadas:v1";

export interface SavedRecipe {
  id: string;
  savedAt: number;
  recipe: Recipe;
  source: SourceInfo | null;
}

export async function loadRecipes(): Promise<SavedRecipe[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    const list = raw ? (JSON.parse(raw) as SavedRecipe[]) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export async function getRecipe(id: string): Promise<SavedRecipe | null> {
  return (await loadRecipes()).find((r) => r.id === id) ?? null;
}

export async function saveRecipe(item: SavedRecipe): Promise<void> {
  const list = (await loadRecipes()).filter((r) => r.id !== item.id);
  await AsyncStorage.setItem(KEY, JSON.stringify([item, ...list]));
}

export async function deleteRecipe(id: string): Promise<void> {
  const list = (await loadRecipes()).filter((r) => r.id !== id);
  await AsyncStorage.setItem(KEY, JSON.stringify(list));
}

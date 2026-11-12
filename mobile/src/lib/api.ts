// Cliente del servidor de análisis (carpeta backend/).
// En un teléfono, "localhost" no sirve: pon en mobile/.env la IP de tu Mac, p. ej.
// EXPO_PUBLIC_API_URL=http://192.168.1.50:8000
export const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type JobStatus =
  | "queued"
  | "downloading"
  | "transcribing"
  | "extracting_frames"
  | "analyzing"
  | "done"
  | "error";

export interface Ingredient {
  name: string;
  quantity: string | null;
  unit: string | null;
  notes: string | null;
  estimated: boolean;
}

export interface Step {
  number: number;
  instruction: string;
  duration_minutes: number | null;
  timestamp: string | null;
}

export interface Recipe {
  is_recipe: boolean;
  title: string;
  description: string;
  servings: string | null;
  prep_time_minutes: number | null;
  cook_time_minutes: number | null;
  total_time_minutes: number | null;
  difficulty: "fácil" | "media" | "difícil" | null;
  ingredients: Ingredient[];
  steps: Step[];
  equipment: string[];
  tips: string[];
  uncertainties: string[];
}

export interface SourceInfo {
  url: string;
  title: string;
  uploader: string;
  duration: number;
  transcript_source: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  created_at: number;
  recipe: Recipe | null;
  source: SourceInfo | null;
  error: string | null;
  error_code?: string | null; // "not_recipe": el clasificador cree que no es una receta
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("No pude conectar con el servidor. Revisa tu conexión.");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new Error(detail ?? `El servidor respondió con un error (${res.status}).`);
  }
  return body as T;
}

// force: analizar aunque el clasificador propio diga que el video no es una receta.
export function createJob(text: string, language = "español", force = false): Promise<Job> {
  return request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ text, language, force }) });
}

export function getJob(id: string): Promise<Job> {
  return request<Job>(`/api/jobs/${encodeURIComponent(id)}`);
}

export interface SearchHit {
  id: string;
  score: number;
}

// Búsqueda semántica (embeddings) limitada a las recetas guardadas por el usuario.
export function searchRecipes(query: string, ids: string[], limit = 50): Promise<SearchHit[]> {
  return request<SearchHit[]>("/api/search", { method: "POST", body: JSON.stringify({ query, ids, limit }) });
}

export interface SimilarHit {
  score: number;
  job: Job;
}

export function similarRecipes(id: string, limit = 6): Promise<SimilarHit[]> {
  return request<SimilarHit[]>(`/api/jobs/${encodeURIComponent(id)}/similar?limit=${limit}`);
}

// Envía la receta corregida por el usuario: es el mejor dato para entrenar modelos propios.
export function sendCorrection(id: string, recipe: Recipe): Promise<{ ok: boolean; changed: boolean }> {
  return request(`/api/jobs/${encodeURIComponent(id)}/correction`, {
    method: "POST",
    body: JSON.stringify({ recipe }),
  });
}

const STATUS_ORDER: JobStatus[] = ["queued", "downloading", "transcribing", "extracting_frames", "analyzing"];

export const STATUS_STEPS: { status: JobStatus; label: string }[] = [
  { status: "downloading", label: "Descargando el video" },
  { status: "transcribing", label: "Escuchando lo que dicen" },
  { status: "extracting_frames", label: "Mirando los ingredientes" },
  { status: "analyzing", label: "Escribiendo la receta" },
];

export function statusIndex(status: JobStatus): number {
  return STATUS_ORDER.indexOf(status);
}

export function recipeAsText(recipe: Recipe, source?: SourceInfo | null): string {
  const lines = [recipe.title, "", recipe.description, ""];
  if (recipe.servings) lines.push(`Porciones: ${recipe.servings}`, "");
  lines.push("INGREDIENTES");
  for (const i of recipe.ingredients) {
    const qty = [i.quantity, i.unit].filter(Boolean).join(" ");
    lines.push(`• ${qty ? qty + " " : ""}${i.name}${i.notes ? ` (${i.notes})` : ""}`);
  }
  lines.push("", "PASOS");
  for (const s of recipe.steps) lines.push(`${s.number}. ${s.instruction}`);
  if (recipe.tips.length) {
    lines.push("", "CONSEJOS");
    for (const t of recipe.tips) lines.push(`• ${t}`);
  }
  if (source?.url) lines.push("", `Video: ${source.url}`);
  return lines.join("\n");
}

// Al compartir un video, el sistema abre la app con un enlace interno de expo-share-intent
// ("recetas://dataUrl=..."). Lo mandamos a la pantalla principal, que es la que lee lo compartido.
export function redirectSystemPath({ path }: { path: string; initial: boolean }) {
  try {
    return path.includes("dataUrl=") ? "/" : path;
  } catch {
    return "/";
  }
}

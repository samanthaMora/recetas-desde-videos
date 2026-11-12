import { useColorScheme } from "react-native";

const light = {
  bg: "#FBF3EA",
  card: "#FFFFFF",
  text: "#2B1F17",
  muted: "#7A6A5D",
  border: "#EADCCD",
  accent: "#C4512B",
  accentSoft: "#F6E1D6",
  onAccent: "#FFFFFF",
  danger: "#B3261E",
};

const dark: typeof light = {
  bg: "#1A1411",
  card: "#251D18",
  text: "#F4EAE1",
  muted: "#A8978A",
  border: "#3A2E26",
  accent: "#E0703F",
  accentSoft: "#3B261C",
  onAccent: "#1A1411",
  danger: "#F2B8B5",
};

export type Theme = typeof light;

export function useTheme(): Theme {
  return useColorScheme() === "dark" ? dark : light;
}

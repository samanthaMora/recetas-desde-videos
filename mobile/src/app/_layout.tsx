import { Stack } from "expo-router";
import { ShareIntentProvider } from "expo-share-intent";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { useTheme } from "@/lib/theme";

export default function RootLayout() {
  const theme = useTheme();

  return (
    <ShareIntentProvider>
      <SafeAreaProvider>
        <StatusBar style="auto" />
        <Stack
          screenOptions={{
            headerStyle: { backgroundColor: theme.bg },
            headerTintColor: theme.accent,
            headerTitleStyle: { color: theme.text },
            headerShadowVisible: false,
            contentStyle: { backgroundColor: theme.bg },
          }}
        >
          <Stack.Screen name="index" options={{ headerShown: false, title: "Recetas" }} />
          <Stack.Screen name="analizar" options={{ title: "Analizando", headerBackTitle: "Atrás" }} />
          <Stack.Screen name="receta/[id]" options={{ title: "Receta", headerBackTitle: "Atrás" }} />
          <Stack.Screen name="editar/[id]" options={{ title: "Editar receta", headerBackTitle: "Cancelar" }} />
        </Stack>
      </SafeAreaProvider>
    </ShareIntentProvider>
  );
}

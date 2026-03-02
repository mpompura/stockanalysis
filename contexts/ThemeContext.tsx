'use client';

import React, { createContext, useContext, useState } from 'react';
import { Theme, defaultTheme } from '@/lib/theme';

type ThemeContextType = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  updateTheme: (partial: Partial<Theme>) => void;
};

const ThemeContext = createContext<ThemeContextType>({
  theme: defaultTheme,
  setTheme: () => {},
  updateTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(defaultTheme);

  const updateTheme = (partial: Partial<Theme>) => {
    setTheme((prev) => ({ ...prev, ...partial }));
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, updateTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

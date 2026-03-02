export type Theme = {
  background: string;
  panel: string;
  divider: string;
  textPrimary: string;
  textMuted: string;
  bull: string;
  bear: string;
  radius: number;
};

export const defaultTheme: Theme = {
  background: '#0B0D11',
  panel: '#13161E',
  divider: '#1E2332',
  textPrimary: '#E2E8F0',
  textMuted: '#64748B',
  bull: '#22D3A5',
  bear: '#F45B69',
  radius: 8,
};

export const presets: { name: string; theme: Theme }[] = [
  {
    name: 'Default Dark',
    theme: defaultTheme,
  },
  {
    name: 'Midnight Blue',
    theme: {
      ...defaultTheme,
      background: '#030A1C',
      panel: '#0C1530',
      divider: '#1E3A5F',
      textPrimary: '#CBD5E1',
      textMuted: '#4A6A9A',
      bull: '#60A5FA',
      bear: '#F97316',
    },
  },
  {
    name: 'Forest Dark',
    theme: {
      ...defaultTheme,
      background: '#060F06',
      panel: '#0E1F0E',
      divider: '#1C3A1C',
      textPrimary: '#DCFCE7',
      textMuted: '#4D7B4D',
      bull: '#4ADE80',
      bear: '#FBBF24',
    },
  },
  {
    name: 'Deep Purple',
    theme: {
      ...defaultTheme,
      background: '#08050F',
      panel: '#150E22',
      divider: '#2A1A45',
      textPrimary: '#E9D5FF',
      textMuted: '#6B46A0',
      bull: '#A78BFA',
      bear: '#FB7185',
    },
  },
];

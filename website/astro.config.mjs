// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// GitHub Pages config: repo lives at katrinarna/Crowd-Dynamics-2 with the
// website in a `website/` subfolder. The deployed site lives at
// https://katrinarna.github.io/Crowd-Dynamics-2/  so we set base accordingly.
const SITE = 'https://katrinarna.github.io';
const BASE = '/Crowd-Dynamics-2';

// https://astro.build/config
export default defineConfig({
  site: SITE,
  base: BASE,
  trailingSlash: 'ignore',
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});

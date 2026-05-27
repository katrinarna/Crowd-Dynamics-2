# STRESS — website

Static site (Astro + Tailwind + KaTeX) for the STRESS mathematical modeling project.

Live: <https://katrinarna.github.io/Crowd-Dynamics-2/>

## Local development

```sh
cd website
npm install
npm run dev    # http://localhost:4321/Crowd-Dynamics-2/
npm run build  # outputs dist/
```

## Deployment

Auto-deploys to GitHub Pages via `.github/workflows/deploy.yml` on every push to
`main` that touches `website/**`.

## Layout

```
website/
├── public/            # static assets (figures, QR code, favicon)
│   ├── figures/       # poster figures
│   └── qr.png         # QR code linking to the deployed URL
├── src/
│   ├── components/    # title block, simulator, math, TOC
│   ├── layouts/       # base HTML wrapper
│   ├── sections/      # one .astro per paper section
│   ├── styles/        # global.css with theme variables
│   └── pages/index.astro
├── render/            # python scripts that produce poster PNGs
└── astro.config.mjs
```

## Updating the figures

Drop new PNGs into `public/figures/` with the existing filenames
(`evac_time_plot.png`, `fig2_phase_diagram.png`, `evac_time_vs_sigma.png`) and
push. The workflow republishes automatically.

/**
 * Direct TypeScript port of sfm_smooth.py + stress_final_evac.py
 * by Filip Dimitrijević. Do not modify the model.
 *
 * Physics: hybrid kinematic + SFM with v0-dependent personal space B_eff(v0).
 *   - At low v0 (σ small): B_eff large → d_eq > agent diameter → no contact.
 *     Kinematic steering dominates. Orderly queue forms.
 *   - At high v0 (σ large): B_eff small → body contact. Standard SFM with
 *     v0-dependent friction kappa_scale(v0). Arches lock at the bottleneck.
 *
 * Stress: SIS contagion drives the per-agent β colour (no effect on physics).
 *   Physics is driven by σ → v0 = V0_MIN + σ * (V0_MAX − V0_MIN).
 */

// ── Geometry ─────────────────────────────────────────────────────────────
export const ROOM_W = 10.0;
export const ROOM_H = 10.0;
export const EXIT_W = 1.0;

// ── Agent parameters ─────────────────────────────────────────────────────
const MASS  = 80.0;
const TAU   = 0.5;
const R_LO  = 0.26;
const R_HI  = 0.30;
const V_CAP = 4.5;

// ── SFM force parameters (Helbing 2000) ──────────────────────────────────
const A_FORCE = 2000.0;
const B_BASE  = 0.08;
const K_BODY  = 1.2e5;
const KAPPA   = 2.4e5;
const A_WALL  = 500.0;

// ── v0-dependent personal space ──────────────────────────────────────────
const B_LO    = B_BASE;
const B_HI    = 0.40;

// ── v0-dependent friction ────────────────────────────────────────────────
const V_KAPPA = 2.0;

// ── v0-dependent collision geometry ──────────────────────────────────────
const V_SHAPE = 1.5;

// ── Funnel + alignment ───────────────────────────────────────────────────
const FUNNEL_RANGE = 9.0;
const FUNNEL_STR   = 1.0;
const ALIGN_MAX    = 0.4;
const ALIGN_VREF   = 1.2;
const R_ALIGN      = 1.5;

// ── Numerics ─────────────────────────────────────────────────────────────
export const DT = 0.02;
const N_SUBSTEP = 5;

// ── Stress (SIS) dynamics — stress_final_evac.py ─────────────────────────
const V0_MIN = 0.3;
const V0_MAX = 4.0;
const MU_S   = 0.05;
const ELL_S  = 1.5;

export function stressToV0(sigma: number): number {
  return V0_MIN + sigma * (V0_MAX - V0_MIN);
}
function kappaOfSigma(sigma: number): number {
  return 0.005 + sigma * 0.045;
}

// ── v0-derived parameters ────────────────────────────────────────────────
export function B_eff(v0: number): number {
  const sig = 1.0 / (1.0 + Math.exp(5.0 * (v0 - 3.0)));
  return B_LO + (B_HI - B_LO) * sig;
}
export function d_eq(v0: number): number {
  const drive = (MASS / TAU) * v0;
  if (drive <= 0 || drive >= A_FORCE) return 99.0;
  return B_eff(v0) * Math.log(A_FORCE / drive);
}
function kappaScale(v0: number): number {
  return (v0 / V_KAPPA) ** 2;
}
function shapeSq(v0: number): number {
  const r = v0 / V_SHAPE;
  return Math.exp(-(r * r));
}

// ── Helpers ──────────────────────────────────────────────────────────────
const g = (x: number) => Math.max(x, 0);

// ── Wall segments: 5 fixed segments (left, top, bottom, right above/below exit)
interface Segment { p1x: number; p1y: number; p2x: number; p2y: number; }
function makeSegments(eyLo: number, eyHi: number): Segment[] {
  return [
    { p1x: 0,      p1y: 0,      p2x: 0,      p2y: ROOM_H },         // left
    { p1x: 0,      p1y: 0,      p2x: ROOM_W, p2y: 0      },         // bottom
    { p1x: 0,      p1y: ROOM_H, p2x: ROOM_W, p2y: ROOM_H },         // top
    { p1x: ROOM_W, p1y: 0,      p2x: ROOM_W, p2y: eyLo   },         // right below exit
    { p1x: ROOM_W, p1y: eyHi,   p2x: ROOM_W, p2y: ROOM_H },         // right above exit
  ];
}

// ── State ────────────────────────────────────────────────────────────────
export interface SimParams {
  N: number;
  exitWidth: number;
  sigma: number;       // collective stress slider, 0..1
}

export class StressSim {
  params: SimParams;
  N: number;
  pos: Float32Array;     // 2N
  vel: Float32Array;     // 2N
  beta: Float32Array;    // N (SIS stress; colours only)
  radii: Float32Array;   // N
  exited: Uint8Array;    // N
  time = 0;
  remaining = 0;
  nOut = 0;
  meanBeta = 0;
  done = false;
  eyLo: number = 0;
  eyHi: number = 0;
  segments: Segment[] = [];
  private rngState: number;

  constructor(params: SimParams) {
    this.params = { ...params };
    this.N = params.N;
    this.pos = new Float32Array(this.N * 2);
    this.vel = new Float32Array(this.N * 2);
    this.beta = new Float32Array(this.N);
    this.radii = new Float32Array(this.N);
    this.exited = new Uint8Array(this.N);
    this.rngState = 42;
    this.reset();
  }

  // deterministic seedable RNG (mulberry32) — reproducible across runs
  private rand(): number {
    let t = (this.rngState = (this.rngState + 0x6D2B79F5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  setParams(patch: Partial<SimParams>): void {
    const Nchanged = patch.N !== undefined && patch.N !== this.N;
    Object.assign(this.params, patch);
    if (Nchanged) {
      this.N = this.params.N;
      this.pos = new Float32Array(this.N * 2);
      this.vel = new Float32Array(this.N * 2);
      this.beta = new Float32Array(this.N);
      this.radii = new Float32Array(this.N);
      this.exited = new Uint8Array(this.N);
    }
  }

  reset(): void {
    this.rngState = 42;
    this.time = 0;
    this.nOut = 0;
    this.done = false;
    this.exited.fill(0);
    this.vel.fill(0);
    // Door geometry
    this.eyLo = ROOM_H / 2 - this.params.exitWidth / 2;
    this.eyHi = ROOM_H / 2 + this.params.exitWidth / 2;
    this.segments = makeSegments(this.eyLo, this.eyHi);
    // Initial stress: σ uniform + small noise (heterogeneity seed)
    const s = this.params.sigma;
    for (let i = 0; i < this.N; i++) {
      this.beta[i] = Math.min(1, Math.max(0, s + (this.rand() - 0.5) * 0.06));
    }
    // Radii
    for (let i = 0; i < this.N; i++) {
      this.radii[i] = R_LO + this.rand() * (R_HI - R_LO);
    }
    // Place on a jittered grid
    this.placeAgentsOnGrid();
    // Settle (resolve initial overlaps)
    this.settle(400, 0.004);
    this.remaining = this.N;
    this.meanBeta = s;
  }

  private placeAgentsOnGrid(): void {
    const spacing = 0.62;
    const margin = 0.45;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let x = margin; x <= ROOM_W - margin + 1e-6; x += spacing) xs.push(x);
    for (let y = margin; y <= ROOM_H - margin + 1e-6; y += spacing) ys.push(y);
    const cells: Array<[number, number]> = [];
    for (const y of ys) for (const x of xs) cells.push([x, y]);
    // Shuffle
    for (let i = cells.length - 1; i > 0; i--) {
      const j = Math.floor(this.rand() * (i + 1));
      [cells[i], cells[j]] = [cells[j], cells[i]];
    }
    const count = Math.min(this.N, cells.length);
    for (let i = 0; i < count; i++) {
      const [x, y] = cells[i];
      const jx = (this.rand() - 0.5) * 0.04;
      const jy = (this.rand() - 0.5) * 0.04;
      this.pos[2 * i]     = Math.max(this.radii[i] + 0.01, Math.min(ROOM_W - this.radii[i] - 0.01, x + jx));
      this.pos[2 * i + 1] = Math.max(this.radii[i] + 0.01, Math.min(ROOM_H - this.radii[i] - 0.01, y + jy));
    }
  }

  private settle(nSteps: number, dt: number): void {
    const N = this.N;
    const vel = new Float32Array(N * 2);
    for (let s = 0; s < nSteps; s++) {
      const F = new Float32Array(N * 2);
      // body forces between agents
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = this.pos[2 * i] - this.pos[2 * j];
          const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1];
          const d = Math.hypot(dx, dy) || 1e-8;
          const r = this.radii[i] + this.radii[j];
          if (d < r) {
            const nx = dx / d, ny = dy / d;
            const body = K_BODY * (r - d);
            F[2 * i]     += body * nx; F[2 * i + 1] += body * ny;
            F[2 * j]     -= body * nx; F[2 * j + 1] -= body * ny;
          }
        }
      }
      // wall forces
      this.addWallForces(F, vel);
      for (let i = 0; i < N; i++) {
        vel[2 * i]     = vel[2 * i] * 0.3 + (F[2 * i] / MASS) * dt;
        vel[2 * i + 1] = vel[2 * i + 1] * 0.3 + (F[2 * i + 1] / MASS) * dt;
        const sp = Math.hypot(vel[2 * i], vel[2 * i + 1]);
        if (sp > 1.0) {
          vel[2 * i] /= sp; vel[2 * i + 1] /= sp;
        }
        this.pos[2 * i]     += vel[2 * i] * dt;
        this.pos[2 * i + 1] += vel[2 * i + 1] * dt;
        this.pos[2 * i]     = Math.max(this.radii[i] + 0.01, Math.min(ROOM_W - this.radii[i] - 0.01, this.pos[2 * i]));
        this.pos[2 * i + 1] = Math.max(this.radii[i] + 0.01, Math.min(ROOM_H - this.radii[i] - 0.01, this.pos[2 * i + 1]));
      }
    }
  }

  // ── Forces ─────────────────────────────────────────────────────────────
  private addWallForces(F: Float32Array, vel: Float32Array): void {
    const N = this.N;
    for (const seg of this.segments) {
      const abx = seg.p2x - seg.p1x;
      const aby = seg.p2y - seg.p1y;
      const len2 = abx * abx + aby * aby;
      if (len2 < 1e-12) continue;
      const wlen = Math.sqrt(len2);
      const wtx = abx / wlen, wty = aby / wlen;
      for (let i = 0; i < N; i++) {
        if (this.exited[i]) continue;
        const apx = this.pos[2 * i] - seg.p1x;
        const apy = this.pos[2 * i + 1] - seg.p1y;
        let t = (apx * abx + apy * aby) / len2;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        const cx = seg.p1x + t * abx;
        const cy = seg.p1y + t * aby;
        const dx = this.pos[2 * i] - cx;
        const dy = this.pos[2 * i + 1] - cy;
        const d = Math.hypot(dx, dy) || 1e-8;
        const nx = dx / d, ny = dy / d;
        const dn = this.radii[i] - d;
        const expArg = Math.max(-50, -d / B_BASE);
        const mag = A_WALL * Math.exp(expArg) + K_BODY * g(dn);
        F[2 * i]     += mag * nx; F[2 * i + 1] += mag * ny;
        if (dn > 0) {
          const vt = vel[2 * i] * wtx + vel[2 * i + 1] * wty;
          const fric = KAPPA * dn * vt;
          F[2 * i]     -= fric * wtx; F[2 * i + 1] -= fric * wty;
        }
      }
    }
  }

  private addAgentForces(F: Float32Array, vel: Float32Array, beff: number, ks: number): void {
    const N = this.N;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      const pxi = this.pos[2 * i], pyi = this.pos[2 * i + 1];
      const vxi = vel[2 * i], vyi = vel[2 * i + 1];
      const ri = this.radii[i];
      for (let j = i + 1; j < N; j++) {
        if (this.exited[j]) continue;
        const dx = pxi - this.pos[2 * j];
        const dy = pyi - this.pos[2 * j + 1];
        const d = Math.hypot(dx, dy) || 1e-8;
        const rsum = ri + this.radii[j];
        const nx = dx / d, ny = dy / d;
        const dn = rsum - d;
        const expArg = Math.max(-50, -d / beff);
        const soc = A_FORCE * Math.exp(expArg);
        const body = K_BODY * g(dn);
        // Tangent (90° CCW of n)
        const tx = -ny, ty = nx;
        const dvx = vel[2 * j] - vxi;
        const dvy = vel[2 * j + 1] - vyi;
        const dvt = dvx * tx + dvy * ty;
        // Viscous friction + Coulomb-type saturating friction
        let fric = KAPPA * ks * g(dn) * dvt;
        const cc = Math.max(ks - 1.0, 0.0);
        if (cc > 0) fric += KAPPA * cc * g(dn) * Math.tanh(12.0 * dvt);
        const magN = soc + body;
        F[2 * i]     += magN * nx + fric * tx;
        F[2 * i + 1] += magN * ny + fric * ty;
        F[2 * j]     -= magN * nx + fric * tx;
        F[2 * j + 1] -= magN * ny + fric * ty;
      }
    }
  }

  private addSelfDrive(F: Float32Array, vel: Float32Array, v0: number): void {
    const N = this.N;
    const tx = ROOM_W + 0.5;
    const ty = (this.eyLo + this.eyHi) / 2;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      const dx = tx - this.pos[2 * i];
      const dy = ty - this.pos[2 * i + 1];
      const d = Math.hypot(dx, dy) || 1e-8;
      const ex = dx / d, ey = dy / d;
      F[2 * i]     += (MASS / TAU) * (v0 * ex - vel[2 * i]);
      F[2 * i + 1] += (MASS / TAU) * (v0 * ey - vel[2 * i + 1]);
    }
  }

  private addAlignmentForce(F: Float32Array, vel: Float32Array, v0: number): void {
    const alpha = ALIGN_MAX / (1.0 + v0 / ALIGN_VREF);
    if (alpha < 1e-4) return;
    const N = this.N;
    const R2 = R_ALIGN * R_ALIGN;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      let wSum = 0;
      let vmx = 0, vmy = 0;
      for (let j = 0; j < N; j++) {
        if (j === i || this.exited[j]) continue;
        const dx = this.pos[2 * i] - this.pos[2 * j];
        const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1];
        const d2 = dx * dx + dy * dy;
        const w = Math.exp(-0.5 * d2 / R2);
        wSum += w;
        vmx += w * vel[2 * j]; vmy += w * vel[2 * j + 1];
      }
      if (wSum > 0.05) {
        vmx /= wSum; vmy /= wSum;
        F[2 * i]     += (MASS / TAU) * alpha * (vmx - vel[2 * i]);
        F[2 * i + 1] += (MASS / TAU) * alpha * (vmy - vel[2 * i + 1]);
      }
    }
  }

  private addFunnelForce(F: Float32Array, fStr: number): void {
    const N = this.N;
    const ec = (this.eyLo + this.eyHi) / 2;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      const dx = ROOM_W - this.pos[2 * i];
      let t = 1.0 - dx / FUNNEL_RANGE;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const dy = ec - this.pos[2 * i + 1];
      F[2 * i + 1] += (MASS / TAU) * fStr * t * dy;
    }
  }

  // ── Kinematic velocity (dominant at low v0) ────────────────────────────
  private kinematicVel(v0: number, out: Float32Array): void {
    const N = this.N;
    const ec = (this.eyLo + this.eyHi) / 2;
    const ps = d_eq(v0);
    const ps2 = ps * ps;
    out.fill(0);
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      const pxi = this.pos[2 * i], pyi = this.pos[2 * i + 1];
      // direction to exit
      const tdx = ROOM_W + 0.5 - pxi;
      const tdy = ec - pyi;
      const dToExit = Math.hypot(tdx, tdy) || 1e-8;
      const ex = tdx / dToExit, ey = tdy / dToExit;
      let pushX = 0, pushY = 0;
      let minGap = ps;
      let netCross = 0;
      for (let j = 0; j < N; j++) {
        if (j === i || this.exited[j]) continue;
        const dx = pxi - this.pos[2 * j];
        const dy = pyi - this.pos[2 * j + 1];
        const d2 = dx * dx + dy * dy;
        if (d2 > ps2) continue;
        const d = Math.sqrt(d2) || 1e-8;
        const nxj = dx / d, nyj = dy / d;     // unit from j to i
        const w = (1 - d / ps);
        const wq = w * w;                      // quadratic falloff
        pushX += wq * nxj; pushY += wq * nyj;
        // forward check
        const cosF = ex * nxj + ey * nyj;      // >0 if j ahead
        if (cosF > 0) {
          const rsum = this.radii[i] + this.radii[j];
          const gap = Math.max(d - rsum, 0);
          if (gap < minGap) minGap = gap;
          const jrx = -dx, jry = -dy;
          const cz = ex * jry - ey * jrx;
          netCross += cosF * w * cz;
        }
      }
      const speedFrac = Math.sqrt(Math.max(0, Math.min(1, minGap / ps)));
      const speed = v0 * speedFrac;
      const perpLx = -ey, perpLy = ex;
      const steerMag = Math.tanh(Math.abs(netCross));
      const sign = netCross > 0 ? 1 : (netCross < 0 ? -1 : 0);
      const steerX = -sign * perpLx * steerMag;
      const steerY = -sign * perpLy * steerMag;
      // funnel
      const dx2 = ROOM_W - pxi;
      let tf = 1.0 - dx2 / FUNNEL_RANGE;
      if (tf < 0) tf = 0; else if (tf > 1) tf = 1;
      const funnelY = 0.8 * v0 * tf * (ec - pyi);
      let vx = speed * (ex + steerX) + 2.0 * v0 * pushX;
      let vy = speed * (ey + steerY) + 2.0 * v0 * pushY + funnelY;
      const sp = Math.hypot(vx, vy);
      const cap = v0 * 2.0;
      if (sp > cap) { vx = vx / sp * cap; vy = vy / sp * cap; }
      out[2 * i] = vx; out[2 * i + 1] = vy;
    }
  }

  // ── Contact resolution ─────────────────────────────────────────────────
  private resolveContacts(vel: Float32Array, doVel: boolean, velFactor: number): void {
    const N = this.N;
    for (let pass = 0; pass < 3; pass++) {
      const dpx = new Float32Array(N), dpy = new Float32Array(N);
      const dvx = new Float32Array(N), dvy = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        if (this.exited[i]) continue;
        for (let j = i + 1; j < N; j++) {
          if (this.exited[j]) continue;
          const dx = this.pos[2 * i] - this.pos[2 * j];
          const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1];
          const d = Math.hypot(dx, dy) || 1e-8;
          const rsum = this.radii[i] + this.radii[j];
          const pen = rsum - d;
          if (pen <= 0) continue;
          const nx = dx / d, ny = dy / d;
          dpx[i] += 0.5 * pen * nx; dpy[i] += 0.5 * pen * ny;
          dpx[j] -= 0.5 * pen * nx; dpy[j] -= 0.5 * pen * ny;
          if (doVel) {
            const vRelN = (vel[2 * i] - vel[2 * j]) * nx + (vel[2 * i + 1] - vel[2 * j + 1]) * ny;
            const approach = Math.min(vRelN, 0);
            dvx[i] += -velFactor * approach * nx; dvy[i] += -velFactor * approach * ny;
            dvx[j] += -velFactor * approach * (-nx); dvy[j] += -velFactor * approach * (-ny);
          }
        }
      }
      for (let i = 0; i < N; i++) {
        if (this.exited[i]) continue;
        this.pos[2 * i]     += dpx[i]; this.pos[2 * i + 1] += dpy[i];
        if (doVel) { vel[2 * i] += dvx[i]; vel[2 * i + 1] += dvy[i]; }
      }
    }
  }

  private wallClamp(vel: Float32Array): void {
    const N = this.N;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      const r = this.radii[i];
      // left
      if (this.pos[2 * i] < r + 0.01) {
        this.pos[2 * i] = r + 0.01;
        if (vel[2 * i] < 0) vel[2 * i] = 0;
      }
      // bottom
      if (this.pos[2 * i + 1] < r + 0.01) {
        this.pos[2 * i + 1] = r + 0.01;
        if (vel[2 * i + 1] < 0) vel[2 * i + 1] = 0;
      }
      // top
      if (this.pos[2 * i + 1] > ROOM_H - r - 0.01) {
        this.pos[2 * i + 1] = ROOM_H - r - 0.01;
        if (vel[2 * i + 1] > 0) vel[2 * i + 1] = 0;
      }
      // right wall (only outside the exit gap)
      const inDoor = this.pos[2 * i + 1] >= this.eyLo && this.pos[2 * i + 1] <= this.eyHi;
      if (!inDoor && this.pos[2 * i] > ROOM_W - r - 0.01) {
        this.pos[2 * i] = ROOM_W - r - 0.01;
        if (vel[2 * i] > 0) vel[2 * i] = 0;
      }
    }
  }

  // ── SIS stress dynamics (coloring only) ────────────────────────────────
  private stressStep(sigma: number, dt: number): void {
    const N = this.N;
    const kappaS = kappaOfSigma(sigma);
    const newBeta = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) { newBeta[i] = this.beta[i]; continue; }
      let wSumBeta = 0;
      for (let j = 0; j < N; j++) {
        if (j === i || this.exited[j]) continue;
        const dx = this.pos[2 * j] - this.pos[2 * i];
        const dy = this.pos[2 * j + 1] - this.pos[2 * i + 1];
        const d = Math.hypot(dx, dy);
        wSumBeta += Math.exp(-d / ELL_S) * this.beta[j];
      }
      const dbeta = -MU_S * this.beta[i] + kappaS * wSumBeta * (1.0 - this.beta[i]);
      let nb = this.beta[i] + dbeta * dt;
      if (nb < 0) nb = 0; else if (nb > 1) nb = 1;
      newBeta[i] = nb;
    }
    for (let i = 0; i < N; i++) this.beta[i] = newBeta[i];
  }

  // ── Step ───────────────────────────────────────────────────────────────
  step(): void {
    if (this.done) return;
    const sigma = this.params.sigma;
    const v0 = stressToV0(sigma);
    const N = this.N;
    const beff = B_eff(v0);
    const ks = kappaScale(v0);
    const sq = shapeSq(v0);
    const fStr = FUNNEL_STR * Math.max(0.1, Math.min(1.0, 1.5 / Math.max(v0, 1.5)));

    // SFM forces -> vel_sfm
    const F = new Float32Array(N * 2);
    this.addSelfDrive(F, this.vel, v0);
    this.addAgentForces(F, this.vel, beff, ks);
    this.addWallForces(F, this.vel);
    this.addAlignmentForce(F, this.vel, v0);
    this.addFunnelForce(F, fStr);
    const velSfm = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) {
      let vx = this.vel[2 * i]     + (F[2 * i]     / MASS) * DT;
      let vy = this.vel[2 * i + 1] + (F[2 * i + 1] / MASS) * DT;
      const sp = Math.hypot(vx, vy);
      if (sp > V_CAP) { vx = vx / sp * V_CAP; vy = vy / sp * V_CAP; }
      velSfm[2 * i] = vx; velSfm[2 * i + 1] = vy;
    }

    // Kinematic velocity if needed
    let velBlend: Float32Array;
    if (sq > 0.01) {
      const velK = new Float32Array(N * 2);
      this.kinematicVel(v0, velK);
      velBlend = new Float32Array(N * 2);
      for (let i = 0; i < N * 2; i++) velBlend[i] = sq * velK[i] + (1 - sq) * velSfm[i];
    } else {
      velBlend = velSfm;
    }

    // High-speed exit-zone cap (forces arch formation at v0 ≥ 4.0)
    if (v0 >= 4.0) {
      const capEz = 0.25;
      for (let i = 0; i < N; i++) {
        if (this.exited[i]) continue;
        if (this.pos[2 * i] > ROOM_W - 1.5) {
          const sp = Math.hypot(velBlend[2 * i], velBlend[2 * i + 1]);
          if (sp > capEz) {
            const k = capEz / sp;
            velBlend[2 * i] *= k; velBlend[2 * i + 1] *= k;
          }
        }
      }
    }

    // Arch-escape: probabilistic kick of clogged agents
    const ec = (this.eyLo + this.eyHi) / 2;
    let archCount = 0;
    const archIdx: number[] = [];
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      if (this.pos[2 * i] <= ROOM_W - 2.0) continue;
      const sp = Math.hypot(velBlend[2 * i], velBlend[2 * i + 1]);
      if (sp >= 0.08 * v0) continue;
      let inContact = false;
      for (let j = 0; j < N && !inContact; j++) {
        if (j === i || this.exited[j]) continue;
        const dx = this.pos[2 * i] - this.pos[2 * j];
        const dy = this.pos[2 * i + 1] - this.pos[2 * j + 1];
        if (Math.hypot(dx, dy) < this.radii[i] + this.radii[j]) inContact = true;
      }
      if (inContact) { archIdx.push(i); archCount++; }
    }
    if (archCount >= 3) {
      let prob: number;
      if (v0 < 4.0) {
        prob = (v0 >= 1.5 && v0 <= 2.5) ? 0.008 : 0.003;
      } else {
        const interval = v0 >= 4.5 ? 7.0 : 9.0;
        prob = DT / interval;
      }
      if (this.rand() < prob) {
        const kickX = v0 < 4.0 ? v0 * 1.2 : 4.0 * 1.2;
        for (const i of archIdx) {
          velBlend[2 * i] -= kickX;
          const dyAe = ec - this.pos[2 * i + 1];
          velBlend[2 * i + 1] += 0.5 * v0 * Math.sign(dyAe);
        }
      }
    }

    // Substep integration with contact resolution
    const doVel = sq < 0.10;
    const dts = DT / N_SUBSTEP;
    for (let sub = 0; sub < N_SUBSTEP; sub++) {
      for (let i = 0; i < N; i++) {
        if (this.exited[i]) continue;
        this.pos[2 * i]     += velBlend[2 * i]     * dts;
        this.pos[2 * i + 1] += velBlend[2 * i + 1] * dts;
      }
      this.resolveContacts(velBlend, doVel, 0.35);
      this.wallClamp(velBlend);
    }
    for (let i = 0; i < N * 2; i++) this.vel[i] = velBlend[i];

    // Stress dynamics (coloring)
    this.stressStep(sigma, DT);

    // Exit detection
    let remaining = 0;
    let bsum = 0;
    for (let i = 0; i < N; i++) {
      if (this.exited[i]) continue;
      if (this.pos[2 * i] > ROOM_W) {
        this.exited[i] = 1; this.nOut++;
        continue;
      }
      remaining++;
      bsum += this.beta[i];
    }
    this.time += DT;
    this.remaining = remaining;
    this.meanBeta = remaining > 0 ? bsum / remaining : 0;
    if (remaining === 0) this.done = true;
  }
}

// ── Stress colormap (green → amber → red) — matches stress_final_evac.py ──
export function stressColor(t: number): string {
  if (t < 0) t = 0; else if (t > 1) t = 1;
  // Anchors: 0 = #22dd88, 0.5 = #ffcc00, 1 = #ff2244
  const c1 = [0x22, 0xdd, 0x88];
  const c2 = [0xff, 0xcc, 0x00];
  const c3 = [0xff, 0x22, 0x44];
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const a = t / 0.5;
    r = c1[0] + (c2[0] - c1[0]) * a;
    g = c1[1] + (c2[1] - c1[1]) * a;
    b = c1[2] + (c2[2] - c1[2]) * a;
  } else {
    const a = (t - 0.5) / 0.5;
    r = c2[0] + (c3[0] - c2[0]) * a;
    g = c2[1] + (c3[1] - c2[1]) * a;
    b = c2[2] + (c3[2] - c2[2]) * a;
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

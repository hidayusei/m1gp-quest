import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { getEnemyDisplay, getEnemyImage } from "./enemyConfig";

type MotionSample = {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  accelerationIncludingGravityX: number | null;
  accelerationIncludingGravityY: number | null;
  accelerationIncludingGravityZ: number | null;
  magnitude: number;
};

type PermissionState = "unknown" | "granted" | "denied" | "unsupported";
type ConnectionState = "idle" | "recording" | "sending" | "ok" | "error";
type AppMode = "battle" | "training" | "tutorial" | "phone";
type GameVariant = "game1" | "game2";
type SoundEffectName = "cast-start" | "cast-confirm" | "spiral" | "star" | "z" | "stun" | "enemy-hit" | "enemy-defeat" | "game-over" | "game-clear";
type CastControlMode = "trigger" | "tap";
type BattlePhase = "player" | "countdown" | "input" | "resolving" | "enemy" | "clear" | "gameover";
type ElementType = "fire" | "water" | "thunder" | "heal";
type EnemyActionType = "attack" | "heavy_attack" | "idle" | "debuff" | "charge";
type ShapeLabel = "circle" | "star" | "triangle" | "square" | "zigzag" | "none";
type MotionCode = ShapeLabel;
type TrainingLabel = ShapeLabel;

type Skill = {
  name: string;
  displayName: string;
  combo: MotionCode[];
  element: ElementType;
  power: number;
  score: number;
  effect: "damage" | "heal";
  exactOnly?: boolean;
  mpCost?: number;
  battleEffect?: "damage" | "stun";
};

type Enemy = {
  name: string;
  maxHp: number;
  weakness: ElementType | "none";
  resistances: ElementType[];
  actions: EnemyAction[];
  visual: string;
  image: string;
  background: string;
};

type EnemyAction = {
  type: EnemyActionType;
  damage?: number;
};

type EnemyActionPlan = EnemyAction & {
  id: string;
};

type RecognitionSegment = {
  segment_id: number;
  start_time: number;
  end_time: number;
  duration: number;
  peak_magnitude: number;
  peak_delta: number;
  dominant_axis: string;
  code: MotionCode;
  strength: string;
  repeats: number;
};

type RecognitionResult = {
  ok: boolean;
  sequence: MotionCode[];
  recognizedSequence?: MotionCode[];
  segments: RecognitionSegment[];
  sampleCount: number;
  serverSamples?: number;
  threshold: number;
  baseline: number;
  message: string;
  selectedSkill?: {
    displayName: string;
    combo?: MotionCode[];
    shape?: ShapeLabel;
    element?: ElementType | "none";
    power?: number;
    score: number;
  };
  recognizedShape?: ShapeLabel;
  confidence?: number;
  inputState?: string;
};

type MotionMetrics = {
  autoTriggerMode?: string;
  triggerState?: string;
  triggerAxis?: string;
  rawX?: number;
  rawY?: number;
  rawZ?: number;
  dynamicX?: number;
  dynamicY?: number;
  dynamicZ?: number;
  dynamicMagnitude?: number;
  rangeX?: number;
  rangeY?: number;
  rangeZ?: number;
  jerkX?: number;
  jerkY?: number;
  jerkZ?: number;
  positivePeak?: number;
  negativePeak?: number;
  triggerPeakToPeak?: number;
  triggerDirectionChange?: number;
  triggerImpulse?: number;
  preStableMs?: number;
  postStableMs?: number;
  preStableMagMean?: number;
  preStableJerkMax?: number;
  postStableMagMean?: number;
  postStableJerkMax?: number;
  yDirectionChange?: number;
  zDirectionChange?: number;
  yImpulse?: number;
  zImpulse?: number;
  triggerGatePassed?: boolean;
  strictGatePassed?: boolean;
  triggerCandidateExists?: boolean;
  triggerCandidateDuration?: number;
  finalTriggerDecision?: string;
  triggerModelConfidence?: number;
  triggerModelStatus?: string;
  triggerModelLabel?: string;
  scoreX?: number;
  scoreY?: number;
  scoreZ?: number;
  selectedCommand?: string;
  rejectedReason?: string;
  modelStatus?: string;
  modelLabel?: string;
  modelConfidence?: number;
  dtwTop1Label?: string;
  dtwTop1Distance?: number;
  dtwTop2Label?: string;
  dtwTop2Distance?: number;
  dtwConfidence?: number;
  dtwDistanceThreshold?: number;
  dtwMargin?: number;
  dtwTemplateCount?: number;
  dtwRejectedReason?: string;
  usedModel?: string;
  message?: string;
};

type TrainingStatus = {
  counts: Record<TrainingLabel, number>;
  triggerCounts: Record<"trigger" | "none" | "legacy_P_as_trigger" | "legacy_none", number>;
  modelExists: boolean;
  modelPath: string;
  modelStatus: string;
  modelLabels?: string[];
  triggerModelExists?: boolean;
  triggerModelStatus?: string;
  triggerModelPath?: string;
};

type ResolvedSkill = Skill & {
  matchedCombo: MotionCode[];
  source: "motion" | "debug" | "fallback";
};

type TurnResult = {
  skill: string;
  amount: string;
  detail: string;
};

type BattleEffect = {
  id: number;
  kind: "skill" | "enemy-hit" | "weakness-hit" | "player-hit" | "defeat" | "stage" | "gameover" | "clear" | "miss";
  element?: ElementType;
  label?: string;
};

const SEND_INTERVAL_MS = 500;
const MAX_BATCH_SIZE = 80;
const MAX_QUEUE_SIZE = 1200;
const MAX_LOG_LINES = 12;
const MOTION_INPUT_SECONDS = 12;
const TRAINING_RECORD_SECONDS = 12;
const PLAYER_MAX_HP = 220;
const PLAYER_MAX_MP = 5;
const ENEMY_RESPONSE_DELAY_MS = 2100;
const SOUND_EFFECT_PATHS: Record<SoundEffectName, string> = {
  "cast-start": "/audio/cast-start.mp3",
  "cast-confirm": "/audio/cast-confirm.mp3",
  spiral: "/audio/spiral.mp3",
  star: "/audio/star.mp3",
  z: "/audio/z.mp3",
  stun: "/audio/stun.mp3",
  "enemy-hit": "/audio/enemy-hit.mp3",
  "enemy-defeat": "/audio/enemy-defeat.mp3",
  "game-over": "/audio/game-over.mp3",
  "game-clear": "/audio/game-clear.mp3",
};
const TRAINING_LABELS: TrainingLabel[] = ["circle", "star", "zigzag"];
const TRIGGER_LABELS: ("trigger" | "none")[] = ["trigger", "none"];
const ELEMENT_EMOJI: Record<ElementType, string> = {
  fire: "🔥",
  water: "💧",
  thunder: "⚡",
  heal: "💚",
};

const ENEMIES: Enemy[] = [
  {
    name: "enemy1",
    maxHp: 20,
    weakness: "fire",
    resistances: ["water", "thunder"],
    actions: [
      { type: "attack", damage: 30 },
      { type: "idle" },
      { type: "attack", damage: 40 },
    ],
    visual: "enemy1",
    image: "/images/enemy1.png",
    background: "/images/stage-grassland-v2.png",
  },
  {
    name: "enemy2",
    maxHp: 60,
    weakness: "water",
    resistances: ["fire", "thunder"],
    actions: [
      { type: "heavy_attack", damage: 50 },
      { type: "attack", damage: 30 },
      { type: "debuff" },
    ],
    visual: "enemy2",
    image: "/images/enemy2.png",
    background: "/images/stage-forest-v2.png",
  },
  {
    name: "enemy3",
    maxHp: 60,
    weakness: "thunder",
    resistances: ["fire", "water"],
    actions: [
      { type: "debuff" },
      { type: "attack", damage: 35 },
      { type: "idle" },
      { type: "heavy_attack", damage: 55 },
    ],
    visual: "enemy3",
    image: "/images/enemy3.png",
    background: "/images/stage-crystal-cave.png",
  },
  {
    name: "enemy4",
    maxHp: 150,
    weakness: "water",
    resistances: ["fire", "thunder"],
    actions: [
      { type: "charge" },
      { type: "heavy_attack", damage: 70 },
      { type: "attack", damage: 40 },
      { type: "idle" },
    ],
    visual: "enemy4",
    image: "/images/enemy4.png",
    background: "/images/stage-final-dungeon.png",
  },
];

const DAMAGE_ELEMENTS: Array<Exclude<ElementType, "heal">> = ["water", "fire", "thunder"];

const SKILLS: Skill[] = [
  {
    name: "circle",
    displayName: "Water Spiral",
    combo: ["circle"],
    element: "water",
    power: 50,
    score: 50,
    effect: "damage",
  },
  {
    name: "star",
    displayName: "Fire Star",
    combo: ["star"],
    element: "fire",
    power: 50,
    score: 50,
    effect: "damage",
  },
  {
    name: "zigzag",
    displayName: "Thunder Z",
    combo: ["zigzag"],
    element: "thunder",
    power: 50,
    score: 50,
    effect: "damage",
  },
];

const GAME2_SKILLS: Skill[] = [
  {
    name: "circle",
    displayName: "spiral",
    combo: ["circle"],
    element: "water",
    power: 20,
    score: 20,
    effect: "damage",
    mpCost: 0,
    battleEffect: "damage",
  },
  {
    name: "star",
    displayName: "star",
    combo: ["star"],
    element: "fire",
    power: 60,
    score: 60,
    effect: "damage",
    mpCost: 3,
    battleEffect: "damage",
  },
  {
    name: "zigzag",
    displayName: "z",
    combo: ["zigzag"],
    element: "thunder",
    power: 0,
    score: 50,
    effect: "damage",
    mpCost: 2,
    battleEffect: "stun",
  },
];

const FALLBACK_SKILL: Skill = {
  name: "none",
  displayName: "No Skill",
  combo: ["none"],
  element: "water",
  power: 0,
  score: 0,
  effect: "damage",
};

const DEBUG_SEQUENCES: { label: string; sequence: MotionCode[] }[] = [
  { label: "circle", sequence: ["circle"] },
  { label: "star", sequence: ["star"] },
  { label: "zigzag", sequence: ["zigzag"] },
];

const TUTORIAL_STEPS: Array<
  | {
      kind: "trigger";
      title: string;
      shortTitle: string;
      prompt: string;
      detail: string;
    }
  | {
      kind: "shape";
      label: TrainingLabel;
      title: string;
      shortTitle: string;
      prompt: string;
      detail: string;
    }
> = [
  {
    kind: "trigger",
    title: "Cast Trigger",
    shortTitle: "Trigger",
    prompt: "Push the phone forward, pause briefly, then return.",
    detail: "This starts casting. The same gesture confirms the cast after drawing.",
  },
  {
    kind: "shape",
    label: "circle",
    title: "Spiral Motion",
    shortTitle: "Spiral",
    prompt: "Draw a big swirl in the air.",
    detail: "Smooth circular motion. This becomes Water Spiral.",
  },
  {
    kind: "shape",
    label: "star",
    title: "Star Motion",
    shortTitle: "Star",
    prompt: "Draw a star shape with sharp corners.",
    detail: "Make the corners clear. This becomes Fire Star.",
  },
  {
    kind: "shape",
    label: "zigzag",
    title: "Z Motion",
    shortTitle: "Z",
    prompt: "Draw a large Z shape.",
    detail: "Move horizontally, diagonally, then horizontally. This becomes Thunder Z.",
  },
];

function initialMode(): AppMode {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "training") {
    return "training";
  }
  if (params.get("mode") === "tutorial") {
    return "tutorial";
  }
  return params.get("mode") === "phone" ? "phone" : "battle";
}

function finiteNumber(value: number | null | undefined): number {
  return Number.isFinite(value) ? Number(value) : 0;
}

function formatValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(3);
}

function formatMetric(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "-";
}

function formatSequence(sequence: MotionCode[]): string {
  return sequence.length > 0 ? sequence.join(" → ") : "-";
}

function visibleSequence(sequence: MotionCode[]): MotionCode[] {
  return sequence.filter((code) => code !== "none");
}

function mergeMotionSequences(previous: MotionCode[], incoming: MotionCode[]): MotionCode[] {
  const current = visibleSequence(previous);
  const next = visibleSequence(incoming);
  if (next.length === 0) {
    return current;
  }
  if (current.length === 0) {
    return next.slice(-16);
  }

  const currentStartsWithNext = next.every((code, index) => current[index] === code);
  if (currentStartsWithNext && next.length <= current.length) {
    return current;
  }

  const nextStartsWithCurrent = current.every((code, index) => next[index] === code);
  if (nextStartsWithCurrent) {
    return next.slice(-16);
  }

  const maxOverlap = Math.min(current.length, next.length);
  for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
    const suffix = current.slice(current.length - overlap);
    const prefix = next.slice(0, overlap);
    if (suffix.every((code, index) => code === prefix[index])) {
      return [...current, ...next.slice(overlap)].slice(-16);
    }
  }

  return [...current, ...next].slice(-16);
}

function formatPlayerSequence(sequence: MotionCode[]): string {
  const visible = visibleSequence(sequence);
  return visible.length > 0 ? visible.map(shapeLabelText).join(" / ") : "図形なし";
}

function elementLabel(element: ElementType | "none"): string {
  return element === "none" ? "none" : `${ELEMENT_EMOJI[element]} ${element}`;
}

function resistanceLabel(enemy: Enemy): string {
  return enemy.resistances.map(elementLabel).join("\n");
}

function skillLabel(skill: Skill): string {
  if (skill.power <= 0) {
    return skill.displayName;
  }
  return `${skill.displayName} ${ELEMENT_EMOJI[skill.element]}`;
}

function formatTurnResult(result: TurnResult): string {
  if (result.skill === "-" && result.amount === "-") {
    return result.detail;
  }
  if (result.amount === "-") {
    return `${result.skill}   ${result.detail}`;
  }
  return `${result.skill}   ${result.amount}   ${result.detail}`;
}

function motionLabel(code: MotionCode): string {
  return shapeLabelText(code);
}

function trainingLabelText(label: TrainingLabel): string {
  return shapeLabelText(label);
}

function tutorialShapeHint(label: TrainingLabel): string {
  if (label === "circle") {
    return "Spiral: 渦巻のように回す";
  }
  if (label === "star") {
    return "Star: 星を描く";
  }
  return "Z: 左右に折り返してZを描く";
}

function shapeLabelText(label: ShapeLabel): string {
  const labels: Record<ShapeLabel, string> = {
    circle: "Spiral",
    star: "Star / 星",
    triangle: "unused",
    square: "unused",
    zigzag: "Z",
    none: "none",
  };
  return labels[label];
}

function isMotionCode(value: unknown): value is MotionCode {
  return (
    value === "circle" ||
    value === "star" ||
    value === "triangle" ||
    value === "square" ||
    value === "zigzag" ||
    value === "none"
  );
}

function percent(value: number, max: number): string {
  return `${Math.max(0, Math.min(100, (value / max) * 100))}%`;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function makeEnemyAction(type: EnemyActionType, damage?: number): EnemyActionPlan {
  return {
    type,
    damage,
    id: `${type}-${damage ?? 0}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  };
}

function enemyActionChoices(enemyIndex: number): EnemyAction[] {
  const bossIndex = ENEMIES.length - 1;
  return enemyIndex === bossIndex
    ? [
        { type: "heavy_attack", damage: 50 },
        { type: "idle" },
        { type: "heavy_attack", damage: 150 },
      ]
    : [
        { type: "attack", damage: 30 },
        { type: "heavy_attack", damage: 50 },
        { type: "idle" },
      ];
}

function shuffleEnemyActions(actions: EnemyAction[]): EnemyAction[] {
  const shuffled = [...actions];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// シャッフルバッグ方式: 3種類の行動をすべて使い切るまで同じ行動は出ない
function shuffledActionBag(enemyIndex: number, previous?: EnemyAction | null): EnemyActionPlan[] {
  const bag = shuffleEnemyActions(enemyActionChoices(enemyIndex));
  // ボスの150は前バッグ末尾との連続を禁止
  if (previous?.damage === 150 && (bag[0]?.damage ?? 0) === 150) {
    const swapIndex = 1 + Math.floor(Math.random() * (bag.length - 1));
    [bag[0], bag[swapIndex]] = [bag[swapIndex], bag[0]];
  }
  return bag.map((action) => makeEnemyAction(action.type, action.damage));
}

function randomEnemyAction(enemyIndex: number, previous?: EnemyAction | null): EnemyActionPlan {
  // キューが空のときのフォールバック専用(通常はシャッフルバッグから供給)
  const choices = enemyActionChoices(enemyIndex);
  const filtered = previous?.damage === 150
    ? choices.filter((action) => action.damage !== 150)
    : choices;
  const action = filtered[Math.floor(Math.random() * filtered.length)] ?? { type: "idle" };
  return makeEnemyAction(action.type, action.damage);
}

function createEnemyActionQueue(enemyIndex: number, seedPrevious?: EnemyAction | null): EnemyActionPlan[] {
  const bag = shuffledActionBag(enemyIndex, seedPrevious);
  if (enemyIndex === ENEMIES.length - 1) {
    // ボスの初手は必ず Heavy 50
    const heavyIndex = bag.findIndex((action) => action.type === "heavy_attack" && action.damage === 50);
    if (heavyIndex > 0) {
      const [heavy] = bag.splice(heavyIndex, 1);
      bag.unshift(heavy);
    }
  }
  return bag;
}

function advanceEnemyActionQueue(enemyIndex: number, queue: EnemyActionPlan[]): EnemyActionPlan[] {
  const used = queue[0] ?? randomEnemyAction(enemyIndex);
  let remaining = queue.slice(1);
  if (remaining.length < 2) {
    const previous = remaining[remaining.length - 1] ?? used;
    remaining = [...remaining, ...shuffledActionBag(enemyIndex, previous)];
  }
  return remaining;
}

function enemyActionAt(enemy: Enemy, turnIndex: number): EnemyAction {
  return enemy.actions[turnIndex % enemy.actions.length] ?? { type: "idle" };
}

function enemyActionText(action: EnemyAction, charged = false): string {
  const damage = action.damage ?? 0;
  const chargedDamage = charged && damage > 0 ? Math.round(damage * 1.5) : damage;
  if (action.type === "heavy_attack") {
    return charged ? `Charged Heavy ${chargedDamage}` : `Heavy ${damage}`;
  }
  if (action.type === "attack") {
    return charged ? `Charged Attack ${chargedDamage}` : `Attack ${damage}`;
  }
  if (action.type === "debuff") {
    return "Debuff";
  }
  if (action.type === "charge") {
    return "Charge";
  }
  return "Idle";
  if (action.type === "heavy_attack") {
    return charged ? `ため強攻撃 ${chargedDamage}` : `強攻撃 ${damage}`;
  }
  if (action.type === "attack") {
    return charged ? `ため攻撃 ${chargedDamage}` : `攻撃 ${damage}`;
  }
  if (action.type === "debuff") {
    return "弱体化";
  }
  if (action.type === "charge") {
    return "ためる";
  }
  return "待機";
}

function phaseText(phase: BattlePhase): string {
  return {
    player: "READY",
    countdown: "COUNTDOWN",
    input: "CASTING",
    resolving: "JUDGING",
    enemy: "ENEMY TURN",
    clear: "CLEAR",
    gameover: "GAME OVER",
  }[phase];
  if (phase === "player") {
    return "入力待ち";
  }
  if (phase === "countdown") {
    return "準備";
  }
  if (phase === "input") {
    return "入力中";
  }
  if (phase === "resolving") {
    return "判定中";
  }
  if (phase === "enemy") {
    return "敵行動";
  }
  if (phase === "clear") {
    return "CLEAR";
  }
  return "GAME OVER";
}

function isDangerAction(action: EnemyAction): boolean {
  return (action.damage ?? 0) >= 150;
}

function enemySpeechText(phase: BattlePhase): string {
  if (phase === "player") {
    return "\u6e96\u5099\u306f\u3067\u304d\u305f\u304b\uff1f";
  }
  if (phase === "input") {
    return "\u305d\u306e\u56f3\u5f62\u3092\u898b\u305b\u3066\u307f\u308d\uff01";
  }
  if (phase === "resolving") {
    return "\u898b\u305b\u3066\u3082\u3089\u304a\u3046\u3002";
  }
  if (phase === "enemy") {
    return "\u4eca\u5ea6\u306f\u3053\u3061\u3089\u306e\u756a\u3060\u3002";
  }
  if (phase === "gameover") {
    return "\u307e\u3060\u7acb\u3066\u308b\u304b\uff1f";
  }
  if (phase === "clear") {
    return "\u3053\u308c\u3067\u7d42\u308f\u308a\u3060\u3068\u601d\u3046\u306a\u3088\u3002";
  }
  return "\u2026\u2026";
}

function chooseBestSkill(sequence: MotionCode[], source: "motion" | "debug", skills: Skill[] = SKILLS): ResolvedSkill {
  const shape = sequence.find((code) => code !== "none");
  const skill = skills.find((item) => item.combo[0] === shape);
  if (skill) {
    return { ...skill, matchedCombo: skill.combo, source };
  }
  return { ...FALLBACK_SKILL, matchedCombo: ["none"], source: "fallback" };
}

function damageMultiplier(enemy: Enemy, skill: Skill): number {
  if (skill.element === "heal") {
    return 1;
  }
  return enemy.weakness === skill.element ? 1 : 0;
}

function nextBossWeakness(current: Exclude<ElementType, "heal">): Exclude<ElementType, "heal"> {
  const index = DAMAGE_ELEMENTS.indexOf(current);
  return DAMAGE_ELEMENTS[(index + 1) % DAMAGE_ELEMENTS.length];
}

function isTerminalPhase(value: BattlePhase): boolean {
  return value === "clear" || value === "gameover";
}

function TypewriterText({ text, speed = 42 }: { text: string; speed?: number }) {
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    setVisibleCount(0);
    if (!text) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setVisibleCount((count) => {
        if (count >= text.length) {
          window.clearInterval(timer);
          return count;
        }
        return count + 1;
      });
    }, speed);
    return () => window.clearInterval(timer);
  }, [text, speed]);

  return (
    <span className="typewriter">
      {text.slice(0, visibleCount)}
      {visibleCount >= text.length ? (
        <i className="type-cursor" aria-hidden="true">
          ▼
        </i>
      ) : null}
    </span>
  );
}

function App() {
  const [mode, setMode] = useState<AppMode>(initialMode);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameVariant, setGameVariant] = useState<GameVariant>("game2");
  const [debugOpen, setDebugOpen] = useState(false);
  const [effectsEnabled, setEffectsEnabled] = useState(true);
  const [soundEffectsEnabled, setSoundEffectsEnabled] = useState(true);
  const [bgmEnabled, setBgmEnabled] = useState(true);
  const [battleEffects, setBattleEffects] = useState<BattleEffect[]>([]);
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [battleSessionId, setBattleSessionId] = useState("battle-room");
  const [sensorSessionId, setSensorSessionId] = useState("battle-room");
  const [phoneView, setPhoneView] = useState<"controller" | "play">("controller");
  const [phoneMotionState, setPhoneMotionState] = useState("waiting_for_start");
  const [phoneCastPulse, setPhoneCastPulse] = useState<"start" | "finish" | "">("");
  const [castControlMode, setCastControlMode] = useState<CastControlMode>("trigger");
  const [phoneTapBusy, setPhoneTapBusy] = useState(false);
  const [permission, setPermission] = useState<PermissionState>("unknown");
  const [permissionDetail, setPermissionDetail] = useState("not requested");
  const [isSensorStreaming, setIsSensorStreaming] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [sampleCount, setSampleCount] = useState(0);
  const [sentCount, setSentCount] = useState(0);
  const [queuedSamples, setQueuedSamples] = useState(0);
  const [lastSendStatus, setLastSendStatus] = useState("idle");
  const [lastSendError, setLastSendError] = useState("");
  const [sendTimerActive, setSendTimerActive] = useState(false);
  const [lastFlushSampleCount, setLastFlushSampleCount] = useState(0);
  const [lastFlushTime, setLastFlushTime] = useState("-");
  const [serverSampleCount, setServerSampleCount] = useState(0);
  const [current, setCurrent] = useState<MotionSample>({
    timestamp: 0,
    x: 0,
    y: 0,
    z: 0,
    accelerationIncludingGravityX: null,
    accelerationIncludingGravityY: null,
    accelerationIncludingGravityZ: null,
    magnitude: 0,
  });
  const [sensorLogs, setSensorLogs] = useState<string[]>([]);

  const [phase, setPhase] = useState<BattlePhase>("player");
  const [playerHp, setPlayerHp] = useState(PLAYER_MAX_HP);
  const [playerMp, setPlayerMp] = useState(PLAYER_MAX_MP);
  const [enemyIndex, setEnemyIndex] = useState(0);
  const [enemyHp, setEnemyHp] = useState(ENEMIES[0].maxHp);
  const [bossWeakness, setBossWeakness] = useState<Exclude<ElementType, "heal">>("water");
  const [enemyTurnIndex, setEnemyTurnIndex] = useState(0);
  const [enemyActionQueue, setEnemyActionQueue] = useState<EnemyActionPlan[]>(() => createEnemyActionQueue(0));
  const [enemyCharged, setEnemyCharged] = useState(false);
  const [bossImageStage, setBossImageStage] = useState(0);
  const [enemyStunned, setEnemyStunned] = useState(false);
  const [countdown, setCountdown] = useState(MOTION_INPUT_SECONDS);
  const [countdownLabel, setCountdownLabel] = useState("");
  const [inputWindow, setInputWindow] = useState<{ startedAt: number; endsAt: number } | null>(null);
  const [recognizedSequence, setRecognizedSequence] = useState<MotionCode[]>([]);
  const [recognizedShape, setRecognizedShape] = useState<ShapeLabel | "">("");
  const [recognitionConfidence, setRecognitionConfidence] = useState(0);
  const [recognizedSegments, setRecognizedSegments] = useState<RecognitionSegment[]>([]);
  const [latestCommand, setLatestCommand] = useState("");
  const [latestMetrics, setLatestMetrics] = useState<MotionMetrics>({});
  const [recognitionStatus, setRecognitionStatus] = useState("idle");
  const [selectedSkill, setSelectedSkill] = useState<ResolvedSkill | null>(null);
  const [damageResult, setDamageResult] = useState("No action yet.");
  const [turnResult, setTurnResult] = useState<TurnResult>({
    skill: "-",
    amount: "-",
    detail: "Press input start.",
  });
  const [lastAttackResult, setLastAttackResult] = useState<TurnResult>({
    skill: "No attack yet",
    amount: "-",
    detail: "-",
  });
  const [lastAttackConfidence, setLastAttackConfidence] = useState(0);
  const [battleLog, setBattleLog] = useState<string[]>([
    "Battle started. Open the phone controller and keep sensor streaming.",
  ]);
  const [latestTurnLog, setLatestTurnLog] = useState<string[]>([
    "Press Start Game, then start phone streaming and begin motion input.",
  ]);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    counts: { circle: 0, star: 0, triangle: 0, square: 0, zigzag: 0, none: 0 },
    triggerCounts: { trigger: 0, none: 0, legacy_P_as_trigger: 0, legacy_none: 0 },
    modelExists: false,
    modelPath: "",
    modelStatus: "unknown",
    modelLabels: [],
  });
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainingCountdown, setTrainingCountdown] = useState(0);
  const [trainingActiveLabel, setTrainingActiveLabel] = useState<TrainingLabel | "">("");
  const [triggerActiveLabel, setTriggerActiveLabel] = useState<"trigger" | "none" | "">("");
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialTriggerChecking, setTutorialTriggerChecking] = useState(false);
  const [tutorialTriggerOk, setTutorialTriggerOk] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState("スマホで Start Streaming してから記録してください。");

  const sampleQueueRef = useRef<MotionSample[]>([]);
  const sendTimerRef = useRef<number | null>(null);
  const isStreamingRef = useRef(false);
  const finishingRef = useRef(false);
  const flushingRef = useRef(false);
  const endpointRef = useRef("");
  const sensorSessionIdRef = useRef("battle-room");
  const phoneMotionStateRef = useRef("");
  const battleMotionStateRef = useRef("");
  const phonePulseTimerRef = useRef<number | null>(null);
  const bgmRef = useRef<HTMLAudioElement | null>(null);
  const soundEffectPoolRef = useRef<Partial<Record<SoundEffectName, HTMLAudioElement>>>({});
  const liveSequenceRef = useRef<MotionCode[]>([]);
  const inputArmedRef = useRef(false);
  const appliedShapeRef = useRef("");
  const phaseRef = useRef<BattlePhase>("player");
  const remoteConfirmArmedRef = useRef(false);

  const apiPrefix = useMemo(() => apiBaseUrl.trim().replace(/\/$/, ""), [apiBaseUrl]);
  const endpoint = useMemo(() => (apiPrefix ? `${apiPrefix}/api/logs/batch` : "/api/logs/batch"), [apiPrefix]);
  const phoneUrl = useMemo(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("mode", "phone");
    return url.toString();
  }, []);
  const isSecurePage = window.isSecureContext;
  const isMixedContentRisk = isSecurePage && apiBaseUrl.trim().startsWith("http://");
  const baseEnemy = ENEMIES[enemyIndex];
  const enemy: Enemy = enemyIndex === ENEMIES.length - 1
    ? {
        ...baseEnemy,
        weakness: bossWeakness,
        resistances: DAMAGE_ELEMENTS.filter((element) => element !== bossWeakness),
      }
    : baseEnemy;
  const currentEnemyAction = enemyActionQueue[0] ?? randomEnemyAction(enemyIndex);
  const followingEnemyAction = enemyActionQueue[1] ?? randomEnemyAction(enemyIndex, currentEnemyAction);
  const enemyDisplay = getEnemyDisplay(enemy.name);
  const battleEnemyDialogue = enemyHp <= 0 ? enemyDisplay.dialogue.defeated : enemyDisplay.dialogue.normal;
  const activeSkills = gameVariant === "game2" ? GAME2_SKILLS : SKILLS;
  const stageStyle = { "--stage-bg": `url(${enemy.background})` } as CSSProperties;
  const playerSequence = useMemo(() => visibleSequence(recognizedSequence), [recognizedSequence]);
  const currentBestSkill = useMemo(
    () => chooseBestSkill(playerSequence, "motion", activeSkills),
    [playerSequence, gameVariant],
  );
  const tutorialStep = TUTORIAL_STEPS[Math.min(tutorialStepIndex, TUTORIAL_STEPS.length - 1)];
  const tutorialComplete = tutorialStepIndex >= TUTORIAL_STEPS.length;

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    if (mode !== "battle" || !gameStarted) {
      battleMotionStateRef.current = recognitionStatus;
      return;
    }
    const previous = battleMotionStateRef.current;
    if (previous && previous !== recognitionStatus) {
      if (recognitionStatus === "recording_motion") {
        playSoundEffect("cast-start", 0.52);
      } else if (previous === "recording_motion" && recognitionStatus === "finished") {
        playSoundEffect("cast-confirm", 0.58);
      }
    }
    battleMotionStateRef.current = recognitionStatus;
  }, [recognitionStatus, mode, gameStarted, soundEffectsEnabled]);

  useEffect(() => {
    for (const [name, path] of Object.entries(SOUND_EFFECT_PATHS) as Array<[SoundEffectName, string]>) {
      const audio = new Audio(path);
      audio.preload = "auto";
      soundEffectPoolRef.current[name] = audio;
    }
    return () => {
      for (const audio of Object.values(soundEffectPoolRef.current)) {
        audio?.pause();
      }
      soundEffectPoolRef.current = {};
    };
  }, []);

  useEffect(() => {
    const bgm = bgmRef.current;
    if (!bgm) {
      return undefined;
    }
    bgm.volume = 0.22;
    bgm.loop = true;
    if (!bgmEnabled || mode === "phone") {
      bgm.pause();
      return undefined;
    }
    const startBgm = () => {
      void bgm.play().catch(() => {
        // Browsers may still require another user gesture before audio starts.
      });
    };
    document.addEventListener("pointerdown", startBgm, { once: true });
    return () => document.removeEventListener("pointerdown", startBgm);
  }, [bgmEnabled, mode]);

  function playSoundEffect(name: SoundEffectName, volume = 0.62) {
    if (!soundEffectsEnabled || mode === "phone") {
      return;
    }
    const source = soundEffectPoolRef.current[name];
    if (!source) {
      return;
    }
    const audio = source.cloneNode(true) as HTMLAudioElement;
    audio.volume = volume;
    void audio.play().catch(() => {
      // A user gesture will unlock audio; battle input continues normally.
    });
  }

  function toggleBgm(enabled: boolean) {
    setBgmEnabled(enabled);
    const bgm = bgmRef.current;
    if (!bgm) {
      return;
    }
    if (enabled && mode !== "phone") {
      void bgm.play().catch(() => undefined);
    } else {
      bgm.pause();
    }
  }

  function pauseBattleBgm() {
    bgmRef.current?.pause();
  }

  function restartBattleBgm() {
    const bgm = bgmRef.current;
    if (!bgmEnabled || mode === "phone" || !bgm) {
      return;
    }
    bgm.currentTime = 0;
    void bgm.play().catch(() => undefined);
  }

  function apiPath(path: string): string {
    return apiPrefix ? `${apiPrefix}${path}` : path;
  }

  async function fetchJsonOrThrow(url: string, init?: RequestInit) {
    const response = await fetch(url, init);
    let result: any = {};
    try {
      result = await response.json();
    } catch {
      result = {};
    }
    if (!response.ok || result.ok === false) {
      const serverMessage = result.error ?? result.message ?? response.statusText;
      const serverPath = result.path ? ` serverPath=${result.path}` : "";
      throw new Error(`${init?.method ?? "GET"} ${url} failed: ${response.status} ${response.statusText}. ${serverMessage}${serverPath}`);
    }
    return result;
  }

  function addSensorLog(message: string) {
    const time = new Date().toLocaleTimeString();
    setSensorLogs((prev) => [`[${time}] ${message}`, ...prev].slice(0, MAX_LOG_LINES));
  }

  function addBattleLog(message: string) {
    const time = new Date().toLocaleTimeString();
    setBattleLog((prev) => [`[${time}] ${message}`, ...prev].slice(0, 18));
  }

  function spawnBattleEffect(kind: BattleEffect["kind"], element?: ElementType, label?: string) {
    if (!effectsEnabled) {
      return;
    }
    const id = Date.now() + Math.random();
    const effect = { id, kind, element, label };
    setBattleEffects((prev) => [...prev, effect].slice(-5));
    const durationByKind: Record<BattleEffect["kind"], number> = {
      skill: 3800,
      "enemy-hit": 3400,
      "weakness-hit": 4200,
      "player-hit": 1700,
      defeat: 3600,
      stage: 3900,
      gameover: 3200,
      clear: 3600,
      miss: 1500,
    };
    window.setTimeout(() => {
      setBattleEffects((prev) => prev.filter((item) => item.id !== id));
    }, durationByKind[kind]);
  }

  async function refreshTrainingStatus() {
    try {
      const response = await fetch(apiPath(`/api/training/status?sessionId=${encodeURIComponent(battleSessionId)}`));
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.error ?? result.message ?? `${response.status} ${response.statusText}`);
      }
      setTrainingStatus({
        counts: {
          circle: result.counts?.circle ?? 0,
          star: result.counts?.star ?? 0,
          triangle: result.counts?.triangle ?? 0,
          square: result.counts?.square ?? 0,
          zigzag: result.counts?.zigzag ?? 0,
          none: result.counts?.none ?? 0,
        },
        triggerCounts: {
          trigger: result.triggerCounts?.trigger ?? 0,
          none: result.triggerCounts?.none ?? 0,
          legacy_P_as_trigger: result.triggerCounts?.legacy_P_as_trigger ?? 0,
          legacy_none: result.triggerCounts?.legacy_none ?? 0,
        },
        modelExists: Boolean(result.shapeModelExists ?? result.modelExists),
        modelPath: result.shapeModelPath ?? result.modelPath ?? "",
        modelStatus: result.shapeModelStatus ?? result.modelStatus ?? "unknown",
        modelLabels: result.shapeModelLabels ?? result.modelLabels ?? [],
        triggerModelExists: Boolean(result.triggerModelExists),
        triggerModelStatus: result.triggerModelStatus ?? "unknown",
        triggerModelPath: result.triggerModelPath ?? "",
      });
      const sampleResponse = await fetch(
        apiPath(`/api/motion/status?sessionId=${encodeURIComponent(battleSessionId)}`),
      );
      if (sampleResponse.ok) {
        const sampleResult = await sampleResponse.json();
        setServerSampleCount(sampleResult.serverSamples ?? sampleResult.sampleCount ?? 0);
      }
    } catch (error) {
      setTrainingMessage(`Training status error: ${String(error)}`);
    }
  }

  async function recordTrainingSample(label: TrainingLabel, source: "training" | "tutorial" = "training") {
    if (trainingBusy) {
      return;
    }
    setTrainingBusy(true);
    setTrainingActiveLabel(label);
    setTrainingMessage(`${trainingLabelText(label)} を記録中です。図形を描いたらStopを押してください。`);
    try {
      const startResponse = await fetch(apiPath("/api/training/record/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: battleSessionId,
          label,
          durationSeconds: TRAINING_RECORD_SECONDS,
        }),
      });
      const startResult = await startResponse.json();
      if (!startResponse.ok || !startResult.ok) {
        throw new Error(startResult.error ?? startResult.message ?? `${startResponse.status} ${startResponse.statusText}`);
      }
      setTrainingMessage(`${trainingLabelText(label)} を記録中です。図形を描いたらStopを押してください。`);
    } catch (error) {
      setTrainingMessage(`Record failed: ${String(error)}`);
      setTrainingBusy(false);
      setTrainingActiveLabel("");
      setTrainingCountdown(0);
      if (source === "tutorial") {
        setMode("tutorial");
      }
    }
  }

  async function stopTrainingSample(saveAsLabel?: TrainingLabel) {
    if (!trainingBusy || !trainingActiveLabel) {
      return;
    }
    const originalLabel = trainingActiveLabel as TrainingLabel;
    const label = saveAsLabel ?? originalLabel;
    try {
      const finishResponse = await fetch(apiPath("/api/training/record/finish"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, label, durationSeconds: TRAINING_RECORD_SECONDS }),
      });
      const finishResult = await finishResponse.json();
      if (!finishResponse.ok || !finishResult.ok) {
        throw new Error(finishResult.error ?? finishResult.message ?? `${finishResponse.status} ${finishResponse.statusText}`);
      }

      setTrainingMessage(
        `${trainingLabelText(label)} を保存しました: ${finishResult.sampleCount} samples / ${finishResult.savedPath}`,
      );
      const savedText =
        label === "none"
          ? `Saved as none. Repeat ${trainingLabelText(originalLabel)} if needed.`
          : `Saved as ${trainingLabelText(label)}.`;
      setTrainingMessage(`${savedText} ${finishResult.sampleCount} samples.`);
      if (mode === "tutorial" && label !== "none") {
        setTutorialStepIndex((index) => Math.min(index + 1, TUTORIAL_STEPS.length));
      }
      await refreshTrainingStatus();
    } catch (error) {
      setTrainingMessage(`Stop failed: ${String(error)}`);
    } finally {
      setTrainingBusy(false);
      setTrainingActiveLabel("");
      setTrainingCountdown(0);
    }
  }

  async function recordTriggerSample(label: "trigger" | "none") {
    if (trainingBusy) {
      return;
    }
    setTrainingBusy(true);
    setTriggerActiveLabel(label);
    setTrainingMessage(
      label === "trigger"
        ? "トリガージェスチャーを記録中です。スマホを前に出す動作をしたらStopを押してください。"
        : "トリガーではない動きを記録中です。図形/構える/横移動などを行ったらStopを押してください。",
    );
    try {
      const url = apiPath("/api/training/trigger/start");
      await fetchJsonOrThrow(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, label }),
      });
    } catch (error) {
      setTrainingMessage(`Trigger record failed: ${String(error)}`);
      setTrainingBusy(false);
      setTriggerActiveLabel("");
    }
  }

  async function stopTriggerSample() {
    if (!trainingBusy || !triggerActiveLabel) {
      return;
    }
    const label = triggerActiveLabel;
    try {
      const url = apiPath("/api/training/trigger/finish");
      const result = await fetchJsonOrThrow(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, label }),
      });
      setTrainingMessage(`Trigger ${label} を保存しました: ${result.sampleCount} samples / ${result.savedPath}`);
      await refreshTrainingStatus();
    } catch (error) {
      setTrainingMessage(`Trigger stop failed: ${String(error)}`);
    } finally {
      setTrainingBusy(false);
      setTriggerActiveLabel("");
    }
  }

  async function trainMotionModel() {
    if (trainingBusy) {
      return;
    }
    setTrainingBusy(true);
    setTrainingMessage("Training model...");
    try {
      const response = await fetch(apiPath("/api/training/train"), { method: "POST" });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.error ?? result.message ?? `${response.status} ${response.statusText}`);
      }
      setTrainingMessage(
        `${result.message}. samples=${result.sampleCount ?? "-"} accuracy=${
          typeof result.accuracy === "number" ? result.accuracy.toFixed(3) : "small-data"
        }`,
      );
      await refreshTrainingStatus();
    } catch (error) {
      setTrainingMessage(`Train failed: ${String(error)}`);
      await refreshTrainingStatus();
    } finally {
      setTrainingBusy(false);
    }
  }

  function clearSendTimer() {
    if (sendTimerRef.current !== null) {
      window.clearInterval(sendTimerRef.current);
      sendTimerRef.current = null;
    }
    setSendTimerActive(false);
  }

  async function requestPermission(): Promise<PermissionState> {
    const DeviceMotionEventWithPermission = window.DeviceMotionEvent as typeof DeviceMotionEvent & {
      requestPermission?: () => Promise<PermissionState>;
    };

    if (typeof window.DeviceMotionEvent === "undefined") {
      setPermission("unsupported");
      setPermissionDetail("DeviceMotionEvent is not available");
      addSensorLog("DeviceMotion API is not supported by this browser.");
      return "unsupported";
    }

    if (typeof DeviceMotionEventWithPermission.requestPermission === "function") {
      try {
        const result = await DeviceMotionEventWithPermission.requestPermission();
        const nextPermission = result === "granted" ? "granted" : "denied";
        setPermission(nextPermission);
        setPermissionDetail(`requestPermission() returned ${result}`);
        addSensorLog(`Sensor permission: ${result}`);
        return nextPermission;
      } catch (error) {
        setPermission("denied");
        setPermissionDetail(`requestPermission() failed: ${String(error)}`);
        addSensorLog(`Permission error: ${String(error)}`);
        return "denied";
      }
    }

    setPermission("granted");
    setPermissionDetail("requestPermission() is not required on this browser");
    addSensorLog("DeviceMotionEvent.requestPermission() is not required on this browser.");
    return "granted";
  }

  async function startSensorStream() {
    const nextPermission = permission === "granted" ? "granted" : await requestPermission();
    if (nextPermission === "denied" || nextPermission === "unsupported") {
      addSensorLog("Sensor is not available. Check browser permission/settings.");
      return;
    }
    if (!window.isSecureContext) {
      addSensorLog("This page is not a secure context. iPhone Safari requires HTTPS for motion sensors.");
    }
    sampleQueueRef.current = [];
    isStreamingRef.current = true;
    clearSendTimer();
    sendTimerRef.current = window.setInterval(() => {
      void flushQueuedSamples();
    }, SEND_INTERVAL_MS);
    setSendTimerActive(true);
    setIsSensorStreaming(true);
    setConnection("recording");
    setSampleCount(0);
    setSentCount(0);
    setQueuedSamples(0);
    setLastSendStatus("streaming");
    setLastSendError("");
    setLastFlushSampleCount(0);
    setLastFlushTime("-");
    addSensorLog(`Sensor streaming started for session "${sensorSessionId}".`);
  }

  async function stopSensorStream() {
    isStreamingRef.current = false;
    clearSendTimer();
    setIsSensorStreaming(false);
    if (flushingRef.current) {
      await new Promise((resolve) => {
        window.setTimeout(resolve, SEND_INTERVAL_MS + 100);
      });
    }
    await flushQueuedSamples(true);
    setConnection("idle");
    setLastSendStatus("stopped");
    addSensorLog("Sensor streaming stopped.");
  }

  async function flushQueuedSamples(drain = false) {
    if (flushingRef.current) {
      return;
    }
    if (sampleQueueRef.current.length === 0) {
      setQueuedSamples(0);
      return;
    }

    flushingRef.current = true;
    setConnection("sending");
    let failedBatch: MotionSample[] = [];
    let hadError = false;
    try {
      do {
        const samples = sampleQueueRef.current.splice(0, MAX_BATCH_SIZE);
        failedBatch = samples;
        setQueuedSamples(sampleQueueRef.current.length);
        if (samples.length === 0) {
          break;
        }

        setLastSendStatus(`sending ${samples.length} samples`);
        const response = await fetch(endpointRef.current, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: sensorSessionIdRef.current, samples }),
        });

        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        failedBatch = [];
        setSentCount((count) => count + samples.length);
        setLastSendStatus("OK");
        setLastSendError("");
        setLastFlushSampleCount(samples.length);
        setLastFlushTime(new Date().toLocaleTimeString());
        setConnection(isStreamingRef.current ? "recording" : "ok");
        addSensorLog(`Sent ${samples.length} samples to ${result.file ?? "CSV"}.`);
      } while (drain && sampleQueueRef.current.length > 0);
    } catch (error) {
      hadError = true;
      sampleQueueRef.current = [...failedBatch, ...sampleQueueRef.current].slice(-MAX_QUEUE_SIZE);
      setLastSendStatus("send failed");
      setLastSendError(String(error));
      setConnection("error");
      addSensorLog(`Send failed: ${String(error)}`);
      if (sampleQueueRef.current.length > MAX_QUEUE_SIZE) {
        sampleQueueRef.current = sampleQueueRef.current.slice(-MAX_QUEUE_SIZE);
      }
      setQueuedSamples(sampleQueueRef.current.length);
    } finally {
      flushingRef.current = false;
      if (isStreamingRef.current && !hadError) {
        setConnection("recording");
      }
    }
  }

  async function armShapeInput() {
    if (phase !== "player") {
      return;
    }
    finishingRef.current = false;
    inputArmedRef.current = true;
    appliedShapeRef.current = "";
    liveSequenceRef.current = [];
    setRecognizedSequence([]);
    setRecognizedShape("");
    setRecognitionConfidence(0);
    setRecognizedSegments([]);
    setLatestCommand("");
    setLatestMetrics({});
    setRecognitionStatus("waiting_for_start");
    setSelectedSkill(null);
    setDamageResult("スマホを前に出してキャスト開始");
    setTurnResult({ skill: "-", amount: "-", detail: "キャスト開始" });
    setCountdown(MOTION_INPUT_SECONDS);
    setCountdownLabel("");
    setLatestTurnLog(["スマホを前に出してキャスト開始"]);
    try {
      const response = await fetch(apiPath("/api/motion/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId }),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      setInputWindow(null);
      addBattleLog(`Shape input armed. Session: ${battleSessionId}`);
    } catch (error) {
      setPhase("player");
      setCountdownLabel("");
      setDamageResult(`Input start failed: ${String(error)}`);
      setTurnResult({ skill: "-", amount: "-", detail: "Input start failed" });
      setLatestTurnLog([`入力開始に失敗: ${String(error)}`]);
      addBattleLog(`Input start failed: ${String(error)}`);
    }
  }

  async function finishMotionInput() {
    setPhase("resolving");
    setCountdownLabel("");
    setDamageResult("Recognizing motion...");
    setTurnResult({ skill: "Judging", amount: "-", detail: "Recognizing motion" });
    try {
      const response = await fetch(apiPath("/api/motion/finish"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, durationSeconds: MOTION_INPUT_SECONDS }),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const result = (await response.json()) as RecognitionResult;
      if (isTerminalPhase(phaseRef.current)) {
        return;
      }
      const shape = isMotionCode(result.recognizedShape) ? result.recognizedShape : "none";
      const sequence: MotionCode[] = [shape];
      liveSequenceRef.current = visibleSequence(sequence);
      setRecognizedShape(shape);
      setRecognitionConfidence(result.confidence ?? 0);
      setRecognizedSequence(sequence);
      setRecognizedSegments(result.segments ?? []);
      setLatestCommand(shape);
      setServerSampleCount(result.serverSamples ?? result.sampleCount ?? 0);
      setRecognitionStatus("finished");
      addBattleLog(
        `Recognized ${shape} from ${result.sampleCount} samples. Message: ${result.message}`,
      );
      resolveSequence(sequence, "motion", result.confidence ?? 0);
    } catch (error) {
      setPhase("player");
      setDamageResult(`Recognition failed: ${String(error)}`);
      setTurnResult({ skill: "-", amount: "-", detail: "Recognition failed" });
      setLatestTurnLog([`判定に失敗: ${String(error)}`]);
      addBattleLog(`Recognition failed: ${String(error)}`);
    }
  }

  function resolveSequence(sequence: MotionCode[], source: "motion" | "debug", confidence = recognitionConfidence) {
    if (phase === "clear" || phase === "gameover") {
      return;
    }
    const playerCodes = visibleSequence(sequence);
    const skill = chooseBestSkill(playerCodes, source, activeSkills);
    liveSequenceRef.current = playerCodes;
    setRecognizedSequence(sequence);
    setRecognizedShape(playerCodes[0] ?? "none");
    setRecognitionConfidence(confidence);
    setSelectedSkill(skill);
    applySkill(skill, playerCodes.length > 0 ? playerCodes : sequence, confidence);
  }

  async function changeCastControlMode(nextMode: CastControlMode) {
    try {
      const result = await fetchJsonOrThrow(apiPath("/api/motion/mode"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, mode: nextMode }),
      });
      setCastControlMode(result.controlMode === "tap" ? "tap" : "trigger");
      addBattleLog(`Cast control mode: ${nextMode}.`);
    } catch (error) {
      addBattleLog(`Cast control mode change failed: ${String(error)}`);
    }
  }

  async function handlePhoneWandTap() {
    if (castControlMode !== "tap" || !isSensorStreaming || phoneTapBusy) {
      return;
    }
    setPhoneTapBusy(true);
    try {
      const result = await fetchJsonOrThrow(apiPath("/api/motion/tap"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: sensorSessionId }),
      });
      const nextState = String(result.inputState ?? "waiting_for_start");
      setPhoneMotionState(nextState);
    } catch (error) {
      setLastSendError(`Wand tap failed: ${String(error)}`);
    } finally {
      window.setTimeout(() => setPhoneTapBusy(false), 450);
    }
  }

  function forceDebugSkill(skill: Skill) {
    if (phase === "countdown" || phase === "input" || phase === "resolving" || phase === "enemy") {
      return;
    }
    const resolvedSkill: ResolvedSkill = { ...skill, matchedCombo: skill.combo, source: "debug" };
    liveSequenceRef.current = skill.combo;
    setRecognizedSequence(skill.combo);
    setRecognizedShape(skill.combo[0] ?? "none");
    setRecognitionConfidence(1);
    setSelectedSkill(resolvedSkill);
    addBattleLog(`Debug cast ${skillLabel(skill)}.`);
    applySkill(resolvedSkill, skill.combo, 1);
  }

  function forceDebugGameOver() {
    inputArmedRef.current = false;
    finishingRef.current = true;
    setInputWindow(null);
    spawnBattleEffect("gameover", undefined, "GAME OVER");
    pauseBattleBgm();
    playSoundEffect("game-over", 0.72);
    setPlayerHp(0);
    setPhase("gameover");
    const result = { skill: "GAME OVER", amount: "HP 0", detail: "Debug" };
    setTurnResult(result);
    setLastAttackResult(result);
    setLastAttackConfidence(0);
    setDamageResult("Debug forced GAME OVER.");
    setLatestTurnLog(["Debug forced GAME OVER."]);
    addBattleLog("Debug forced GAME OVER.");
  }

  function forceDebugGameClear() {
    inputArmedRef.current = false;
    finishingRef.current = true;
    setInputWindow(null);
    spawnBattleEffect("clear", undefined, "GAME CLEAR");
    pauseBattleBgm();
    playSoundEffect("game-clear", 0.72);
    setEnemyHp(0);
    setPhase("clear");
    const result = { skill: "GAME CLEAR", amount: "All enemies defeated", detail: "Debug" };
    setTurnResult(result);
    setLastAttackResult(result);
    setLastAttackConfidence(1);
    setDamageResult("Debug forced GAME CLEAR.");
    setLatestTurnLog(["Debug forced GAME CLEAR."]);
    addBattleLog("Debug forced GAME CLEAR.");
  }

  async function trainTriggerModel() {
    if (trainingBusy) {
      return;
    }
    setTrainingBusy(true);
    setTrainingMessage("Training trigger detector...");
    try {
      const url = apiPath("/api/training/trigger/train");
      const result = await fetchJsonOrThrow(url, { method: "POST" });
      setTrainingMessage(
        `${result.message}. samples=${result.sampleCount ?? "-"} accuracy=${
          typeof result.accuracy === "number" ? result.accuracy.toFixed(3) : "small-data"
        }`,
      );
      await refreshTrainingStatus();
    } catch (error) {
      setTrainingMessage(`Trigger train failed: ${String(error)}`);
      await refreshTrainingStatus();
    } finally {
      setTrainingBusy(false);
    }
  }

  async function archiveTriggerTraining() {
    if (trainingBusy) {
      return;
    }
    const ok = window.confirm(
      "現在のTrigger Trainingデータとtrigger modelを退避して、新しく集め直します。削除ではなくarchiveに移動します。実行しますか？",
    );
    if (!ok) {
      return;
    }
    setTrainingBusy(true);
    setTrainingMessage("Archiving trigger training data...");
    try {
      const result = await fetchJsonOrThrow(apiPath("/api/training/trigger/archive"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true }),
      });
      setTrainingMessage(
        `Trigger data archived. trigger=${result.movedCounts?.trigger ?? 0} none=${
          result.movedCounts?.none ?? 0
        } -> ${result.archiveDir}`,
      );
      addBattleLog("Trigger training data archived. Collect new trigger/none samples carefully.");
      await refreshTrainingStatus();
    } catch (error) {
      setTrainingMessage(`Trigger archive failed: ${String(error)}`);
      await refreshTrainingStatus();
    } finally {
      setTrainingBusy(false);
    }
  }

  async function saveLastTriggerCandidate(label: "trigger" | "none") {
    try {
      const url = apiPath("/api/training/trigger/save-candidate");
      const result = await fetchJsonOrThrow(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, label }),
      });
      setTrainingMessage(result.message ?? `Saved last trigger candidate as ${label}.`);
      addBattleLog(`Saved last trigger candidate as ${label}.`);
      await refreshTrainingStatus();
    } catch (error) {
      const message = `Save trigger candidate failed: ${String(error)}`;
      setTrainingMessage(message);
      addBattleLog(message);
    }
  }

  async function saveLastShapeInput(label: TrainingLabel) {
    try {
      const url = apiPath("/api/training/shape/save-last");
      const result = await fetchJsonOrThrow(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId, label }),
      });
      const labelText = trainingLabelText(label);
      setTrainingMessage(result.message ?? `Saved last shape input as ${labelText}.`);
      addBattleLog(`Saved last shape input as ${labelText}.`);
      await refreshTrainingStatus();
    } catch (error) {
      const message = `Save shape input failed: ${String(error)}`;
      setTrainingMessage(message);
      addBattleLog(message);
    }
  }

  function recoverMpAfterTurn() {
    if (gameVariant === "game2") {
      setPlayerMp((current) => Math.min(PLAYER_MAX_MP, current + 1));
    }
  }

  function applyEnemyAction(enemyForTurn: Enemy, hpBeforeAction = playerHp, turnLines: string[] = [], stunOverride = false) {
    setPhase("enemy");
    setDamageResult(`${enemyForTurn.name} is acting...`);
    const action = enemyActionQueue[0] ?? randomEnemyAction(enemyIndex);
    const wasCharged = enemyCharged;

    window.setTimeout(() => {
      if (isTerminalPhase(phaseRef.current)) {
        return;
      }
      setEnemyTurnIndex((turn) => turn + 1);
      setEnemyActionQueue((queue) => advanceEnemyActionQueue(enemyIndex, queue));

      if (gameVariant === "game2" && (enemyStunned || stunOverride) && (action.type === "attack" || action.type === "heavy_attack")) {
        playSoundEffect("stun", 0.66);
        const nextLines = [...turnLines, `${enemyForTurn.name}'s attack was cancelled by Stun.`];
        setEnemyStunned(false);
        setEnemyCharged(false);
        setDamageResult(`${enemyForTurn.name}'s attack was cancelled.`);
        setLatestTurnLog(nextLines);
        addBattleLog(`${enemyForTurn.name}'s attack was cancelled by Stun.`);
        recoverMpAfterTurn();
        inputArmedRef.current = false;
        finishingRef.current = false;
        setPhase("player");
        return;
      }

      if (action.type === "idle") {
        const nextLines = [...turnLines, `${enemyForTurn.name}は様子を見ている`];
        setDamageResult(`${enemyForTurn.name} is waiting.`);
        setLatestTurnLog(nextLines);
        addBattleLog(`${enemyForTurn.name} stayed idle.`);
        recoverMpAfterTurn();
        inputArmedRef.current = false;
        finishingRef.current = false;
        setPhase("player");
        return;
      }

      if (action.type === "charge") {
        const nextLines = [...turnLines, `${enemyForTurn.name}は力をためている`];
        setEnemyCharged(true);
        setDamageResult(`${enemyForTurn.name} charged power.`);
        setLatestTurnLog(nextLines);
        addBattleLog(`${enemyForTurn.name} charged power.`);
        recoverMpAfterTurn();
        inputArmedRef.current = false;
        finishingRef.current = false;
        setPhase("player");
        return;
      }

      if (action.type === "debuff") {
        const nextLines = [
          ...turnLines,
          `${enemyForTurn.name}の弱体化！`,
          "次の与ダメージ x0.8",
        ];
        setDamageResult(`${enemyForTurn.name} used Debuff. Weakness damage is unaffected.`);
        setLatestTurnLog(nextLines);
        addBattleLog(`${enemyForTurn.name} used Debuff.`);
        recoverMpAfterTurn();
        inputArmedRef.current = false;
        finishingRef.current = false;
        setPhase("player");
        return;
      }

      const baseDamage = action.damage ?? 0;
      const damage = wasCharged ? Math.round(baseDamage * 1.5) : baseDamage;
      const nextHp = Math.max(0, hpBeforeAction - damage);
      playSoundEffect("enemy-hit", 0.62);
      spawnBattleEffect("player-hit", undefined, `${damage}`);
      setEnemyCharged(false);
      const nextLines = [
        ...turnLines,
        wasCharged ? `${enemyForTurn.name}のため攻撃！` : `${enemyForTurn.name}の攻撃！`,
        `Playerは${damage}ダメージ`,
      ];
      if (nextHp <= 0) {
        pauseBattleBgm();
        playSoundEffect("game-over", 0.72);
        nextLines.push("Game Over.");
        spawnBattleEffect("gameover", undefined, "GAME OVER");
        const gameOverResult = { skill: "GAME OVER", amount: "HP 0", detail: "Retry?" };
        setTurnResult(gameOverResult);
        setLastAttackResult(gameOverResult);
        setLastAttackConfidence(0);
      }
      setPlayerHp(nextHp);
      setDamageResult(`${enemyForTurn.name} dealt ${damage} damage.`);
      setLatestTurnLog(nextLines);
      addBattleLog(`${enemyForTurn.name} dealt ${damage} damage.`);
      if (nextHp > 0) {
        recoverMpAfterTurn();
      }
      inputArmedRef.current = false;
      finishingRef.current = false;
      setPhase(nextHp <= 0 ? "gameover" : "player");
    }, ENEMY_RESPONSE_DELAY_MS);
  }

  function applySkill(skill: ResolvedSkill, sequence: MotionCode[], confidence: number) {
    const enemyForTurn = enemy;
    const skillName = gameVariant === "game2" ? skill.displayName : skillLabel(skill);
    const shownSequence = formatPlayerSequence(sequence);

    if (gameVariant === "game2" && playerMp < (skill.mpCost ?? 0)) {
      // MP不足: 技エフェクトは出さず MISS 表示のみ
      spawnBattleEffect("miss", undefined, "MISS");
      const turnLines = [`Input: ${shownSequence}`, `${skillName} failed: not enough MP.`];
      const result = { skill: skillName, amount: "0 damage", detail: "Not enough MP" };
      setLatestTurnLog(turnLines);
      setDamageResult(`${skillName} failed: not enough MP.`);
      setTurnResult(result);
      setLastAttackResult(result);
      setLastAttackConfidence(confidence);
      addBattleLog(`${skillName} failed: not enough MP.`);
      applyEnemyAction(enemyForTurn, playerHp, turnLines);
      return;
    }

    spawnBattleEffect("skill", skill.element, skillName);
    addBattleLog(`Sequence ${formatSequence(sequence)} activated ${skillName}.`);
    if (skill.name === "circle") {
      playSoundEffect("spiral", 0.6);
    } else if (skill.name === "star") {
      playSoundEffect("star", 0.64);
    } else if (skill.name === "zigzag") {
      playSoundEffect("z", 0.58);
    }

    if (gameVariant === "game2") {
      const mpCost = skill.mpCost ?? 0;
      setPlayerMp((current) => Math.max(0, current - mpCost));

      if (skill.battleEffect === "stun") {
        const turnLines = [`Input: ${shownSequence}`, "Z activated Stun.", "The enemy's next attack will be cancelled."];
        const result = { skill: skillName, amount: `-${mpCost} MP`, detail: "Stun" };
        setEnemyStunned(true);
        setLatestTurnLog(turnLines);
        setDamageResult("Stun is active. The enemy's next attack will be cancelled.");
        setTurnResult(result);
        setLastAttackResult(result);
        setLastAttackConfidence(confidence);
        addBattleLog(`${skillName} applied Stun to ${enemyForTurn.name}.`);
        applyEnemyAction(enemyForTurn, playerHp, turnLines, true);
        return;
      }
    }

    if (skill.effect === "heal") {
      const nextHp = Math.min(PLAYER_MAX_HP, playerHp + skill.power);
      const healed = nextHp - playerHp;
      const turnLines = [
        `入力：${shownSequence}`,
        `${skillName}が発動！`,
        `Playerは${healed}回復`,
      ];
      setLatestTurnLog(turnLines);
      setPlayerHp(nextHp);
      setDamageResult(`${skillName}: ${healed}回復`);
      const result = { skill: skillName, amount: `+${healed} HP`, detail: "Heal" };
      setTurnResult(result);
      setLastAttackResult(result);
      setLastAttackConfidence(confidence);
      addBattleLog(`${skillName} restored ${healed} HP.`);
      applyEnemyAction(enemyForTurn, nextHp, turnLines);
      return;
    }

    const multiplier = gameVariant === "game2" ? 1 : damageMultiplier(enemyForTurn, skill);
    const isWeaknessHit = gameVariant === "game1" && skill.power > 0 && enemyForTurn.weakness === skill.element;
    const damage =
      multiplier === 0 || skill.power <= 0 ? 0 : Math.max(1, Math.round(skill.power * multiplier));
    const nextEnemyHp = Math.max(0, enemyHp - damage);
    spawnBattleEffect(isWeaknessHit ? "weakness-hit" : "enemy-hit", skill.element);
    setEnemyHp(nextEnemyHp);
    // ボスは攻撃を当てるたびに画像を1段階進める(最後の段階に達したら倒すまでそのまま)
    if (enemyIndex === ENEMIES.length - 1 && damage > 0 && nextEnemyHp > 0) {
      setBossImageStage((stage) => stage + 1);
    }
    if (gameVariant === "game1" && enemyIndex === ENEMIES.length - 1 && damage > 0 && nextEnemyHp > 0) {
      setBossWeakness(nextBossWeakness(bossWeakness));
    }

    const affinity = gameVariant === "game2" ? "Direct hit" : isWeaknessHit ? "Weakness hit" : "Not weakness: no damage";
    const turnLines = [
      `入力：${shownSequence}`,
      `${skillName}が発動！`,
      `${enemyForTurn.name}に${damage}ダメージ`,
      isWeaknessHit ? "弱点攻撃" : "弱点ではないためダメージ0",
    ];
    setLatestTurnLog(turnLines);
    setDamageResult(`${skillName}: ${damage} damage. ${affinity}.`);
    const result = {
      skill: skillName,
      amount: `${damage} damage`,
      detail: gameVariant === "game2" ? `${skill.mpCost ?? 0} MP` : isWeaknessHit ? "Weak" : "Not weak: 0",
    };
    setTurnResult(result);
    setLastAttackResult(result);
    setLastAttackConfidence(confidence);
    addBattleLog(`${skillName} dealt ${damage} damage to ${enemyForTurn.name}. ${affinity}.`);

    if (nextEnemyHp > 0) {
      applyEnemyAction(enemyForTurn, playerHp, turnLines);
      return;
    }

    addBattleLog(`${enemyForTurn.name} defeated.`);
    playSoundEffect("enemy-defeat", 0.72);
    recoverMpAfterTurn();
    spawnBattleEffect("defeat", skill.element);
    const defeatedLines = [...turnLines, `${enemyForTurn.name} defeated.`];
    setLatestTurnLog(defeatedLines);
    if (enemyIndex >= ENEMIES.length - 1) {
      pauseBattleBgm();
      window.setTimeout(() => playSoundEffect("game-clear", 0.78), 650);
      setPhase("clear");
      spawnBattleEffect("clear", undefined, "GAME CLEAR");
      setDamageResult("Game Clear. All enemies defeated.");
      const clearResult = { skill: skillName, amount: `${damage} damage`, detail: "Game Clear" };
      setTurnResult(clearResult);
      setLastAttackResult(clearResult);
      setLastAttackConfidence(confidence);
      setLatestTurnLog([...defeatedLines, "Game Clear. All enemies defeated."]);
      return;
    }

    const nextIndex = enemyIndex + 1;
    const nextEnemy = ENEMIES[nextIndex];
    spawnBattleEffect("stage");
    setPhase("resolving");
    setDamageResult(`${enemyForTurn.name} defeated. ${nextEnemy.name} appears.`);
    const appearResult = { skill: skillName, amount: `${damage} damage`, detail: `${nextEnemy.name} appears` };
    setTurnResult(appearResult);
    setLastAttackResult(appearResult);
    setLastAttackConfidence(confidence);
    addBattleLog(`${nextEnemy.name} appears.`);
    window.setTimeout(() => {
      if (isTerminalPhase(phaseRef.current)) {
        return;
      }
      setEnemyIndex(nextIndex);
      setEnemyHp(nextEnemy.maxHp);
      setEnemyActionQueue(createEnemyActionQueue(nextIndex));
      setBossImageStage(0);
      if (nextIndex === ENEMIES.length - 1) {
        setBossWeakness("water");
      }
      setEnemyTurnIndex(0);
      setEnemyCharged(false);
      setEnemyStunned(false);
      setLatestTurnLog([...defeatedLines, `${nextEnemy.name} appears.`]);
      inputArmedRef.current = false;
      finishingRef.current = false;
      setPhase("player");
    }, 3200);
  }

  function resetBattle(message = "Battle reset.") {
    setPhase("player");
    battleMotionStateRef.current = "";
    inputArmedRef.current = false;
    finishingRef.current = false;
    appliedShapeRef.current = "";
    setPlayerHp(PLAYER_MAX_HP);
    setPlayerMp(PLAYER_MAX_MP);
    setEnemyIndex(0);
    setEnemyHp(ENEMIES[0].maxHp);
    setEnemyActionQueue(createEnemyActionQueue(0));
    setBossImageStage(0);
    setBossWeakness("water");
    setEnemyTurnIndex(0);
    setEnemyCharged(false);
    setEnemyStunned(false);
    setCountdown(MOTION_INPUT_SECONDS);
    setCountdownLabel("");
    setInputWindow(null);
    liveSequenceRef.current = [];
    setRecognizedSequence([]);
    setRecognizedShape("");
    setRecognitionConfidence(0);
    setRecognizedSegments([]);
    setLatestCommand("");
    setLatestMetrics({});
    setRecognitionStatus("idle");
    setSelectedSkill(null);
    setDamageResult(message);
    setTurnResult({ skill: "-", amount: "-", detail: message });
    setLastAttackResult({ skill: "No attack yet", amount: "-", detail: "-" });
    setLastAttackConfidence(0);
    setServerSampleCount(0);
    setBattleLog([message]);
    setLatestTurnLog([message]);
  }

  function startGame() {
    restartBattleBgm();
    resetBattle("Game started. Start phone streaming, then use the trigger gesture to start casting.");
    setGameStarted(true);
    setMode("battle");
  }

  function startTutorial() {
    setMode("tutorial");
    setGameStarted(false);
    setTutorialStepIndex(0);
    setTutorialTriggerChecking(false);
    setTutorialTriggerOk(false);
    setTrainingMessage("Tutorial started. Ask the player to follow the current motion.");
  }

  function resetToStart() {
    restartBattleBgm();
    resetBattle("Back to title.");
    setGameStarted(false);
    setMode("battle");
  }

  async function cancelMotionSession() {
    try {
      await fetchJsonOrThrow(apiPath("/api/motion/cancel"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId }),
      });
    } catch {
      // Tutorial cleanup only. The next start call will replace stale server state if cancel is unavailable.
    }
  }

  async function advanceTutorialStep() {
    if (tutorialStep.kind === "trigger") {
      await cancelMotionSession();
    }
    setTutorialStepIndex((index) => {
      const nextIndex = Math.min(index + 1, TUTORIAL_STEPS.length);
      if (nextIndex !== 0) {
        setTutorialTriggerChecking(false);
        setTutorialTriggerOk(false);
      }
      return nextIndex;
    });
  }

  async function startTutorialTriggerCheck() {
    setTutorialTriggerChecking(true);
    setTutorialTriggerOk(false);
    setTrainingMessage("Waiting for trigger gesture...");
    try {
      await fetchJsonOrThrow(apiPath("/api/motion/start"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId }),
      });
    } catch (error) {
      setTutorialTriggerChecking(false);
      setTrainingMessage(`Trigger check failed: ${String(error)}`);
    }
  }

  async function discardTrainingSample() {
    if (!trainingBusy || !trainingActiveLabel) {
      return;
    }
    try {
      await fetchJsonOrThrow(apiPath("/api/training/record/cancel"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: battleSessionId }),
      });
      setTrainingMessage(`Discarded ${trainingLabelText(trainingActiveLabel as TrainingLabel)} attempt.`);
    } catch (error) {
      setTrainingMessage(`Discard failed: ${String(error)}`);
    } finally {
      setTrainingBusy(false);
      setTrainingActiveLabel("");
      setTrainingCountdown(0);
    }
  }

  useEffect(() => {
    function handleMotion(event: DeviceMotionEvent) {
      const rawX = event.acceleration?.x ?? event.accelerationIncludingGravity?.x;
      const rawY = event.acceleration?.y ?? event.accelerationIncludingGravity?.y;
      const rawZ = event.acceleration?.z ?? event.accelerationIncludingGravity?.z;
      const x = finiteNumber(rawX);
      const y = finiteNumber(rawY);
      const z = finiteNumber(rawZ);
      const magnitude = Math.sqrt(x * x + y * y + z * z);
      const sample: MotionSample = {
        timestamp: Date.now(),
        x,
        y,
        z,
        accelerationIncludingGravityX: event.accelerationIncludingGravity?.x ?? null,
        accelerationIncludingGravityY: event.accelerationIncludingGravity?.y ?? null,
        accelerationIncludingGravityZ: event.accelerationIncludingGravity?.z ?? null,
        magnitude,
      };

      setCurrent(sample);

      if (isStreamingRef.current) {
        sampleQueueRef.current.push(sample);
        if (sampleQueueRef.current.length > MAX_QUEUE_SIZE) {
          sampleQueueRef.current.shift();
        }
        setQueuedSamples(sampleQueueRef.current.length);
        setSampleCount((count) => count + 1);
      }
    }

    window.addEventListener("devicemotion", handleMotion);
    return () => window.removeEventListener("devicemotion", handleMotion);
  }, []);

  useEffect(() => {
    endpointRef.current = endpoint;
    sensorSessionIdRef.current = sensorSessionId;
  }, [endpoint, sensorSessionId]);

  useEffect(() => {
    return () => {
      clearSendTimer();
      if (phonePulseTimerRef.current !== null) {
        window.clearTimeout(phonePulseTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (mode !== "phone" || !isSensorStreaming) {
      phoneMotionStateRef.current = "";
      setPhoneMotionState("waiting_for_start");
      return undefined;
    }

    const pulsePhone = (kind: "start" | "finish") => {
      if ("vibrate" in window.navigator) {
        window.navigator.vibrate(kind === "start" ? [140, 70, 140] : [220, 80, 100]);
      }
      setPhoneCastPulse(kind);
      if (phonePulseTimerRef.current !== null) {
        window.clearTimeout(phonePulseTimerRef.current);
      }
      phonePulseTimerRef.current = window.setTimeout(() => {
        setPhoneCastPulse("");
        phonePulseTimerRef.current = null;
      }, 900);
    };

    const pollMotionState = async () => {
      try {
        const response = await fetch(
          apiPath(`/api/motion/status?sessionId=${encodeURIComponent(sensorSessionId)}`),
        );
        if (!response.ok) {
          return;
        }
        const result = await response.json();
        const nextState = String(result.inputState ?? "waiting_for_start");
        setCastControlMode(result.controlMode === "tap" ? "tap" : "trigger");
        const previousState = phoneMotionStateRef.current;
        setPhoneMotionState(nextState);

        if (previousState && previousState !== nextState) {
          if (nextState === "recording_motion") {
            pulsePhone("start");
          } else if (previousState === "recording_motion") {
            pulsePhone("finish");
          }
        }
        phoneMotionStateRef.current = nextState;
      } catch {
        // Sensor streaming continues even when a status poll temporarily fails.
      }
    };

    void pollMotionState();
    const timer = window.setInterval(() => void pollMotionState(), 250);
    return () => window.clearInterval(timer);
  }, [mode, isSensorStreaming, sensorSessionId, apiPrefix]);

  useEffect(() => {
    if (phase !== "input" || !inputWindow) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      const remaining = Math.max(0, (inputWindow.endsAt - Date.now()) / 1000);
      setCountdown(Math.round(remaining * 10) / 10);
      if (remaining <= 0 && !finishingRef.current) {
        finishingRef.current = true;
        window.clearInterval(timer);
        void finishMotionInput();
      }
    }, 250);
    return () => window.clearInterval(timer);
  }, [phase, inputWindow]);

  useEffect(() => {
    if (mode !== "battle" || !gameStarted || phase !== "player" || inputArmedRef.current) {
      return;
    }
    void armShapeInput();
  }, [mode, gameStarted, phase, battleSessionId]);

  useEffect(() => {
    if (mode !== "battle") {
      return undefined;
    }
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(
          apiPath(`/api/motion/status?sessionId=${encodeURIComponent(battleSessionId)}`),
        );
        if (response.ok) {
          const result = await response.json();
          setCastControlMode(result.controlMode === "tap" ? "tap" : "trigger");
          setServerSampleCount(result.serverSamples ?? result.sampleCount ?? 0);
          if (result.latestMetrics && typeof result.latestMetrics === "object") {
            setLatestMetrics(result.latestMetrics as MotionMetrics);
          }
          const inputState = String(result.inputState ?? "");
          setRecognitionStatus(inputState || "waiting_for_start");
          // タイトル / ゲームオーバー / クリア画面: スマホの杖からの
          // キャスト開始操作(トリガージェスチャー or 杖タップ)をボタン押下として扱う
          if (!gameStarted || isTerminalPhase(phase)) {
            if (
              remoteConfirmArmedRef.current &&
              (inputState === "recording_motion" || inputState === "finished")
            ) {
              // 直後に遷移先の画面が /api/motion/start でセッションを再初期化するため、
              // ここでは cancel を送らずに画面遷移だけ行う
              remoteConfirmArmedRef.current = false;
              if (!gameStarted) {
                startGame();
              } else {
                resetToStart();
              }
            }
            return;
          }
          if (inputState === "waiting_for_start" && phase === "player") {
            setDamageResult("スマホを前に出してキャスト開始");
            setTurnResult({ skill: "-", amount: "-", detail: "キャスト開始" });
          } else if (inputState === "recording_motion") {
            if (phase !== "input") {
              setPhase("input");
            }
            setDamageResult("入力中：平面に図形を描いてください");
            setTurnResult({ skill: "-", amount: "-", detail: "図形を描いて、もう一度スマホを前に出してキャスト確定" });
            if (typeof result.remainingSeconds === "number") {
              setCountdown(Math.round(Math.max(0, result.remainingSeconds) * 10) / 10);
            }
          } else if (
            inputState === "finished" &&
            !finishingRef.current &&
            inputArmedRef.current &&
            (phaseRef.current === "input" || recognitionStatus === "recording_motion")
          ) {
            finishingRef.current = true;
            const shape = isMotionCode(result.recognizedShape) ? result.recognizedShape : "none";
            const confidence = Number(result.confidence ?? 0);
            appliedShapeRef.current = shape;
            setPhase("resolving");
            setRecognizedShape(shape);
            setRecognitionConfidence(confidence);
            setRecognizedSequence([shape]);
            setLatestCommand(shape);
            setSelectedSkill(chooseBestSkill(shape === "none" ? [] : [shape], "motion", activeSkills));
            setTurnResult({
              skill: shapeLabelText(shape),
              amount: `${Math.round(confidence * 100)}%`,
              detail: result.message ?? "recognized",
            });
            resolveSequence(shape === "none" ? ["none"] : [shape], "motion", confidence);
          }
        }
      } catch {
        setServerSampleCount(0);
        setRecognitionStatus("status error");
      }
    }, phase === "input" || phase === "player" ? 450 : 1500);
    return () => window.clearInterval(timer);
  }, [mode, battleSessionId, apiPrefix, phase, gameVariant, gameStarted]);

  // タイトル / ゲームオーバー / クリア画面では入力セッションを待機状態にして、
  // スマホの杖から PRESS START / Title ボタンを押せるようにする(追加仕様)。
  // Trigger Gesture モードはスマホを前に出す動作、Tap Wand モードは杖タップで反応する。
  useEffect(() => {
    remoteConfirmArmedRef.current = false;
    if (mode !== "battle") {
      return undefined;
    }
    const needsRemoteConfirm = !gameStarted || phase === "gameover" || phase === "clear";
    if (!needsRemoteConfirm) {
      return undefined;
    }
    let cancelled = false;
    const armRemoteConfirm = async () => {
      try {
        const response = await fetch(apiPath("/api/motion/start"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: battleSessionId }),
        });
        if (response.ok && !cancelled) {
          remoteConfirmArmedRef.current = true;
        }
      } catch {
        // スマホからの操作は追加手段なので、失敗しても画面上のボタンはそのまま使える。
      }
    };
    void armRemoteConfirm();
    return () => {
      cancelled = true;
    };
  }, [mode, gameStarted, phase, battleSessionId, apiPrefix]);

  useEffect(() => {
    if (mode !== "training" && mode !== "tutorial") {
      return undefined;
    }
    void refreshTrainingStatus();
    const timer = window.setInterval(() => {
      void refreshTrainingStatus();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [mode, apiPrefix]);

  useEffect(() => {
    if (mode !== "tutorial" || tutorialComplete || tutorialStep.kind !== "trigger" || !tutorialTriggerChecking || tutorialTriggerOk) {
      return undefined;
    }
    const timer = window.setInterval(async () => {
      try {
        const result = await fetchJsonOrThrow(
          apiPath(`/api/motion/status?sessionId=${encodeURIComponent(battleSessionId)}`),
        );
        const inputState = String(result.inputState ?? "");
        if (result.latestMetrics && typeof result.latestMetrics === "object") {
          setLatestMetrics(result.latestMetrics as MotionMetrics);
        }
        if (inputState === "recording_motion" || inputState === "finished") {
          setTutorialTriggerOk(true);
          setTutorialTriggerChecking(false);
          setTrainingMessage("OK. Trigger gesture detected.");
          window.clearInterval(timer);
        }
      } catch (error) {
        setTrainingMessage(`Trigger check error: ${String(error)}`);
      }
    }, 350);
    return () => window.clearInterval(timer);
  }, [mode, tutorialComplete, tutorialStep, tutorialTriggerChecking, tutorialTriggerOk, battleSessionId, apiPrefix]);

  return (
    <main className={`app-shell ${mode === "battle" ? "battle-shell" : ""} ${mode === "phone" && phoneView === "play" ? "phone-play-shell" : ""}`}>
      <audio ref={bgmRef} src="/audio/battle-bgm.mp3" preload="auto" loop />
      <section className="top-bar">
        <strong className="mini-title">
          <img src="/images/m1gp-quest-logo.png" alt="M1GP QUEST" />
        </strong>
        <div className="mode-switch">
          <button className={mode === "battle" ? "active" : ""} onClick={() => setMode("battle")}>
            Battle
          </button>
          <button className={mode === "training" ? "active" : ""} onClick={() => setMode("training")}>
            Training
          </button>
          <button className={mode === "phone" ? "active" : ""} onClick={() => setMode("phone")}>
            Phone
          </button>
          {mode === "battle" ? (
            <button className={debugOpen ? "debug-nav-button active" : "debug-nav-button"} onClick={() => setDebugOpen((open) => !open)}>
              Debug
            </button>
          ) : null}
          {mode === "battle" && gameStarted ? (
            <button className="reset-nav-button" onClick={resetToStart}>
              Reset Game
            </button>
          ) : null}
        </div>
      </section>

      {mode === "battle" ? (
        gameStarted ? (
        <section className="battle-layout">
          <aside className="panel stage-panel">
            <p className="eyebrow">Stage</p>
            <h2>
              Stage {enemyIndex + 1} / {ENEMIES.length}
            </h2>
            <div className="stage-current">{enemy.name}</div>
            <div className="enemy-chain">
              {ENEMIES.map((item, index) => (
                <span
                  key={item.name}
                  className={index === enemyIndex ? "active" : index < enemyIndex ? "cleared" : ""}
                >
                  {item.name}
                </span>
              ))}
            </div>
            <div className={`stage-facts ${gameVariant === "game2" ? "game2-facts" : ""}`}>
              <div>
                <span>ENEMY</span>
                <strong>
                  {enemyIndex + 1} / {ENEMIES.length}
                </strong>
              </div>
            </div>
            <div className="stage-skill-list">
              {activeSkills.map((skill) => (
                <div key={`stage-skill-${skill.name}`}>
                  <span>{skill.displayName}</span>
                  <strong>
                    {gameVariant === "game2"
                      ? `${skill.mpCost ?? 0} MP / ${skill.battleEffect === "stun" ? "Stun" : `${skill.power} attack`}`
                      : elementLabel(skill.element)}
                  </strong>
                </div>
              ))}
            </div>
          </aside>

          <div className="battle-main">
            <section className="enemy-stage panel">
              <div className={`enemy-art phase-${phase} ${battleEffects.some((effect) => effect.kind === "player-hit") ? "is-player-hit" : ""}`} style={stageStyle}>
                {effectsEnabled ? (
                  <div className="battle-effects-layer" aria-hidden="true">
                    {battleEffects.map((effect) => (
                      <div
                        className={`battle-effect effect-${effect.kind} effect-${effect.element ?? "neutral"}`}
                        key={effect.id}
                      >
                        <span>{effect.label ?? ""}</span>
                        <i />
                        <i />
                        <i />
                        <i />
                        <i />
                        <i />
                      </div>
                    ))}
                  </div>
                ) : null}
                {phase === "gameover" ? (
                  <div className="game-over-overlay">
                    <img className="battle-end-title" src="/images/game-over.png" alt="GAME OVER" />
                    <span>PLAYER HP 0</span>
                    <button onClick={resetToStart}>
                      Title
                    </button>
                  </div>
                ) : null}
                {phase === "clear" ? (
                  <div className="game-clear-overlay">
                    <img className="battle-end-title" src="/images/game-clear.png" alt="GAME CLEAR" />
                    <span>ALL ENEMIES DEFEATED</span>
                    <button onClick={resetToStart}>
                      Title
                    </button>
                  </div>
                ) : null}
                <div className="cast-guide-card">
                  <strong className={`cast-guide-title ${phase === "input" ? "is-draw" : ""}`}>
                    {phase === "input"
                      ? "DRAW!"
                      : phase === "resolving"
                        ? "JUDGING"
                        : phase === "gameover"
                          ? "GAME OVER"
                          : phase === "clear"
                            ? "GAME CLEAR"
                            : "READY"}
                  </strong>
                  {phase === "input" ? (
                    <div className="draw-gauge" aria-hidden="true">
                      <div style={{ width: percent(countdown, MOTION_INPUT_SECONDS) }} />
                    </div>
                  ) : null}
                </div>
                <div className={`enemy-placeholder enemy-${enemyIndex} has-image ${battleEffects.some((effect) => effect.kind === "enemy-hit" || effect.kind === "weakness-hit") ? "is-hit" : ""} ${battleEffects.some((effect) => effect.kind === "weakness-hit") ? "is-weakness-hit" : ""} ${battleEffects.some((effect) => effect.kind === "defeat") ? "is-defeated" : ""}`}>
                  <img
                    className="enemy-image"
                    src={getEnemyImage(enemy.name, enemyHp, bossImageStage)}
                    alt={enemyDisplay.displayName}
                    onError={(event) => event.currentTarget.classList.add("is-missing")}
                  />
                  <span>{enemy.visual}</span>
                </div>
                <div className={`enemy-info-card ${gameVariant === "game2" ? "game2-info" : ""}`}>
                  <div>
                    <span>Next</span>
                    <strong className={isDangerAction(currentEnemyAction) ? "danger-action" : ""}>
                      {enemyActionText(currentEnemyAction, enemyCharged)}
                    </strong>
                  </div>
                  <div>
                    <span>After</span>
                    <strong className={isDangerAction(followingEnemyAction) ? "danger-action" : ""}>
                      {enemyActionText(followingEnemyAction)}
                    </strong>
                  </div>
                </div>
                <div className="enemy-summary enemy-stage-card">
                  <div>
                    <strong>{enemyDisplay.displayName}</strong>
                  </div>
                  <div>
                    <strong className="resource-inline">HP&nbsp;&nbsp;{enemyHp} / {enemy.maxHp}</strong>
                    <div className="hp-bar enemy-hp">
                      <div style={{ width: percent(enemyHp, enemy.maxHp) }} />
                    </div>
                  </div>
                </div>
                <div className="enemy-speech-card">
                  <TypewriterText text={battleEnemyDialogue} />
                </div>
              </div>
            </section>

            <section className={`input-banner phase-${phase}`}>
              <div className="input-banner-grid">
                <div className="attack-result-card">
                  <strong>{lastAttackResult.skill}</strong>
                  <small className="english-result">
                    Confidence {lastAttackConfidence > 0 ? `${Math.round(lastAttackConfidence * 100)}%` : "-"} /{" "}
                    {lastAttackResult.amount !== "-" ? lastAttackResult.amount : lastAttackResult.detail}
                  </small>
                  <small>
                    信頼度 {lastAttackConfidence > 0 ? `${Math.round(lastAttackConfidence * 100)}%` : "-"} /{" "}
                    {lastAttackResult.amount !== "-" ? lastAttackResult.amount : lastAttackResult.detail}
                  </small>
                </div>
                <div className="player-status-card">
                  <strong>Player</strong>
                  <small className="resource-inline">HP&nbsp;&nbsp;{playerHp} / {PLAYER_MAX_HP}</small>
                  <div className={`hp-bar player-hp ${playerHp <= PLAYER_MAX_HP * 0.3 ? "low" : ""}`}>
                    <div style={{ width: percent(playerHp, PLAYER_MAX_HP) }} />
                  </div>
                  {gameVariant === "game2" ? (
                    <>
                      <small className="resource-inline">MP&nbsp;&nbsp;{playerMp} / {PLAYER_MAX_MP}</small>
                      <div className="hp-bar mp-bar">
                        <div style={{ width: percent(playerMp, PLAYER_MAX_MP) }} />
                      </div>
                    </>
                  ) : null}
                </div>
              </div>
            </section>
          </div>

          <aside className={`battle-side ${debugOpen ? "debug-open" : ""}`}>
            <section className="panel compact skill-panel">
              <h2>Motions</h2>
              <div className="motion-guide-side">
                <div>
                  <strong>◎</strong>
                  <span>Spiral</span>
                </div>
                <div>
                  <strong>★</strong>
                  <span>Star</span>
                </div>
                <div>
                  <strong>Ｚ</strong>
                  <span>Z</span>
                </div>
              </div>
            </section>

            <section className="panel compact skill-table-panel">
              <h2>Skills</h2>
              <div className="skill-list">
                {activeSkills.map((skill) => {
                  const selected = selectedSkill?.name === skill.name;
                  const relation = selected
                    ? "selected"
                    : gameVariant === "game2"
                      ? "normal"
                      : skill.effect === "heal"
                      ? "heal"
                      : enemy.weakness === skill.element
                        ? "weak"
                        : enemy.resistances.includes(skill.element)
                          ? "resist"
                          : "normal";
                  return (
                    <div className={`skill-row skill-${relation}`} key={skill.name}>
                      <strong>{formatSequence(skill.combo)}</strong>
                      <span>
                        {gameVariant === "game2" ? skill.displayName : skillLabel(skill)}
                        <small>
                          {gameVariant === "game2"
                            ? `${skill.mpCost ?? 0} MP / ${skill.battleEffect === "stun" ? "Stun" : `${skill.power} attack`}`
                            : `${skill.element} / power ${skill.power}`}
                          {skill.exactOnly ? " / exact" : ""}
                        </small>
                      </span>
                    </div>
                  );
                })}
              </div>
            </section>

            <details className="panel debug-panel" open={debugOpen}>
              <summary>Debug</summary>
              <div className="debug-action-group">
                <span>Cast start / finish control</span>
                <div className="debug-mode-switch" role="group" aria-label="Cast control mode">
                  <button
                    className={castControlMode === "trigger" ? "active" : ""}
                    onClick={() => void changeCastControlMode("trigger")}
                    disabled={phase === "input"}
                  >
                    Trigger Gesture
                  </button>
                  <button
                    className={castControlMode === "tap" ? "active" : ""}
                    onClick={() => void changeCastControlMode("tap")}
                    disabled={phase === "input"}
                  >
                    Tap Wand
                  </button>
                </div>
              </div>
              <label className="debug-toggle-row">
                <input
                  type="checkbox"
                  checked={effectsEnabled}
                  onChange={(event) => setEffectsEnabled(event.target.checked)}
                />
                Effects enabled
              </label>
              <label className="debug-toggle-row">
                <input
                  type="checkbox"
                  checked={soundEffectsEnabled}
                  onChange={(event) => setSoundEffectsEnabled(event.target.checked)}
                />
                Sound effects
              </label>
              <label className="debug-toggle-row">
                <input
                  type="checkbox"
                  checked={bgmEnabled}
                  onChange={(event) => toggleBgm(event.target.checked)}
                />
                BGM
              </label>
              <div className="field">
                <label htmlFor="battle-session">Battle session</label>
                <input
                  id="battle-session"
                  value={battleSessionId}
                  onChange={(event) => setBattleSessionId(event.target.value)}
                />
              </div>
              <div className="debug-stats">
                <div>
                  <span>input active</span>
                  <strong>{phase === "input" ? "YES" : "NO"}</strong>
                </div>
                <div>
                  <span>server samples</span>
                  <strong>{serverSampleCount}</strong>
                </div>
                <div>
                  <span>status</span>
                  <strong>{recognitionStatus}</strong>
                </div>
                <div>
                  <span>latest</span>
                  <strong>{latestCommand || "-"}</strong>
                </div>
              </div>
              <div className="debug-metrics">
                <div>
                  <span>auto trigger mode</span>
                  <strong>{latestMetrics.autoTriggerMode || "-"}</strong>
                </div>
                <div>
                  <span>trigger model</span>
                  <strong>{latestMetrics.triggerModelStatus || "disabled for now"}</strong>
                </div>
                <div>
                  <span>trigger state</span>
                  <strong>{latestMetrics.triggerState || "-"}</strong>
                </div>
                <div>
                  <span>trigger_axis</span>
                  <strong>{latestMetrics.triggerAxis || "-"}</strong>
                </div>
                <div>
                  <span>raw x/y/z</span>
                  <strong>
                    {formatMetric(latestMetrics.rawX)} / {formatMetric(latestMetrics.rawY)} / {formatMetric(latestMetrics.rawZ)}
                  </strong>
                </div>
                <div>
                  <span>dynamic x/y/z</span>
                  <strong>
                    {formatMetric(latestMetrics.dynamicX)} / {formatMetric(latestMetrics.dynamicY)} / {formatMetric(latestMetrics.dynamicZ)}
                  </strong>
                </div>
                <div>
                  <span>dynamic magnitude</span>
                  <strong>{formatMetric(latestMetrics.dynamicMagnitude)}</strong>
                </div>
                <div>
                  <span>x_range</span>
                  <strong>{formatMetric(latestMetrics.rangeX)}</strong>
                </div>
                <div>
                  <span>y_range</span>
                  <strong>{formatMetric(latestMetrics.rangeY)}</strong>
                </div>
                <div>
                  <span>z_range</span>
                  <strong>{formatMetric(latestMetrics.rangeZ)}</strong>
                </div>
                <div>
                  <span>x_jerk_max</span>
                  <strong>{formatMetric(latestMetrics.jerkX)}</strong>
                </div>
                <div>
                  <span>y_jerk_max</span>
                  <strong>{formatMetric(latestMetrics.jerkY)}</strong>
                </div>
                <div>
                  <span>z_jerk_max</span>
                  <strong>{formatMetric(latestMetrics.jerkZ)}</strong>
                </div>
                <div>
                  <span>trigger_peak_to_peak</span>
                  <strong>{formatMetric(latestMetrics.triggerPeakToPeak)}</strong>
                </div>
                <div>
                  <span>trigger_direction_change</span>
                  <strong>{formatMetric(latestMetrics.triggerDirectionChange)}</strong>
                </div>
                <div>
                  <span>trigger_impulse</span>
                  <strong>{formatMetric(latestMetrics.triggerImpulse)}</strong>
                </div>
                <div>
                  <span>positive peak</span>
                  <strong>{formatMetric(latestMetrics.positivePeak)}</strong>
                </div>
                <div>
                  <span>negative peak</span>
                  <strong>{formatMetric(latestMetrics.negativePeak)}</strong>
                </div>
                <div>
                  <span>pre stable ms</span>
                  <strong>{formatMetric(latestMetrics.preStableMs)}</strong>
                </div>
                <div>
                  <span>post stable ms</span>
                  <strong>{formatMetric(latestMetrics.postStableMs)}</strong>
                </div>
                <div>
                  <span>y/z direction</span>
                  <strong>
                    {formatMetric(latestMetrics.yDirectionChange)} / {formatMetric(latestMetrics.zDirectionChange)}
                  </strong>
                </div>
                <div>
                  <span>y/z impulse</span>
                  <strong>
                    {formatMetric(latestMetrics.yImpulse)} / {formatMetric(latestMetrics.zImpulse)}
                  </strong>
                </div>
                <div>
                  <span>trigger confidence</span>
                  <strong>{formatMetric(latestMetrics.triggerModelConfidence)}</strong>
                </div>
                <div>
                  <span>trigger gate</span>
                  <strong>{latestMetrics.triggerGatePassed === undefined ? "-" : latestMetrics.triggerGatePassed ? "passed" : "rejected"}</strong>
                </div>
                <div>
                  <span>strict gate result</span>
                  <strong>{latestMetrics.strictGatePassed === undefined ? "-" : latestMetrics.strictGatePassed ? "passed" : "rejected"}</strong>
                </div>
                <div>
                  <span>candidate</span>
                  <strong>{latestMetrics.triggerCandidateExists ? "exists" : "none"}</strong>
                </div>
                <div>
                  <span>candidate duration</span>
                  <strong>{formatMetric(latestMetrics.triggerCandidateDuration)} ms</strong>
                </div>
                <div>
                  <span>model prediction</span>
                  <strong>{latestMetrics.triggerModelLabel || "-"}</strong>
                </div>
                <div>
                  <span>final decision</span>
                  <strong>{latestMetrics.finalTriggerDecision || "-"}</strong>
                </div>
                <div>
                  <span>score_x</span>
                  <strong>{formatMetric(latestMetrics.scoreX)}</strong>
                </div>
                <div>
                  <span>score_y</span>
                  <strong>{formatMetric(latestMetrics.scoreY)}</strong>
                </div>
                <div>
                  <span>score_z</span>
                  <strong>{formatMetric(latestMetrics.scoreZ)}</strong>
                </div>
                <div>
                  <span>selected</span>
                  <strong>{latestMetrics.selectedCommand || "-"}</strong>
                </div>
                <div>
                  <span>DTW top1 label</span>
                  <strong>{latestMetrics.dtwTop1Label || "-"}</strong>
                </div>
                <div>
                  <span>top1 distance</span>
                  <strong>{formatMetric(latestMetrics.dtwTop1Distance)}</strong>
                </div>
                <div>
                  <span>DTW top2 label</span>
                  <strong>{latestMetrics.dtwTop2Label || "-"}</strong>
                </div>
                <div>
                  <span>top2 distance</span>
                  <strong>{formatMetric(latestMetrics.dtwTop2Distance)}</strong>
                </div>
                <div>
                  <span>DTW confidence</span>
                  <strong>{formatMetric(latestMetrics.dtwConfidence)}</strong>
                </div>
                <div>
                  <span>used model</span>
                  <strong>{latestMetrics.usedModel || "-"}</strong>
                </div>
                <div>
                  <span>DTW threshold</span>
                  <strong>{formatMetric(latestMetrics.dtwDistanceThreshold)}</strong>
                </div>
                <div>
                  <span>DTW margin</span>
                  <strong>{formatMetric(latestMetrics.dtwMargin)}</strong>
                </div>
                <div className="debug-wide">
                  <span>DTW rejected reason</span>
                  <strong>{latestMetrics.dtwRejectedReason || "-"}</strong>
                </div>
                <div>
                  <span>model</span>
                  <strong>{latestMetrics.modelStatus || trainingStatus.modelStatus || "-"}</strong>
                </div>
                <div>
                  <span>model label</span>
                  <strong>
                    {latestMetrics.modelLabel || "-"}
                    {latestMetrics.modelConfidence ? ` (${latestMetrics.modelConfidence.toFixed(2)})` : ""}
                  </strong>
                </div>
                <div className="debug-wide">
                  <span>rejected reason</span>
                  <strong>{latestMetrics.rejectedReason || "-"}</strong>
                </div>
              </div>
              <p className="hint">Phone URL: {phoneUrl}</p>
              <button onClick={() => void finishMotionInput()} disabled={phase !== "input"}>
                Finish Now
              </button>
              <div className="debug-action-group">
                <span>Force Battle State</span>
                <div className="debug-grid">
                  <button onClick={forceDebugGameOver}>
                    Show GAME OVER
                  </button>
                  <button onClick={forceDebugGameClear}>
                    Show GAME CLEAR
                  </button>
                </div>
              </div>
              <div className="debug-action-group">
                <span>Force Skill</span>
                <div className="debug-grid">
                  {activeSkills.map((skill) => (
                    <button
                      key={`force-skill-${skill.name}`}
                      onClick={() => forceDebugSkill(skill)}
                      disabled={
                        phase === "countdown" ||
                        phase === "input" ||
                        phase === "resolving" ||
                        phase === "enemy"
                      }
                    >
                      {gameVariant === "game2" ? skill.displayName : skillLabel(skill)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="debug-grid">
                <button onClick={() => void saveLastTriggerCandidate("trigger")}>
                  Save last candidate as trigger
                </button>
                <button onClick={() => void saveLastTriggerCandidate("none")}>
                  Save last candidate as none
                </button>
                <button onClick={() => void trainTriggerModel()}>
                  Train Trigger Model
                </button>
                <button onClick={() => void archiveTriggerTraining()}>
                  Archive Trigger Data
                </button>
                <button onClick={() => void refreshTrainingStatus()}>
                  Refresh Training Status
                </button>
              </div>
              <p className="hint">
                直前の図形判定が違うときは、正しいラベルで保存してからTrain Shape Modelを押してください。
              </p>
              <div className="debug-grid">
                {TRAINING_LABELS.map((label) => (
                  <button key={`save-last-shape-${label}`} onClick={() => void saveLastShapeInput(label)}>
                    Save shape as {trainingLabelText(label)}
                  </button>
                ))}
                <button onClick={() => void trainMotionModel()}>
                  Train Shape Model
                </button>
              </div>
              <div className="debug-grid">
                {DEBUG_SEQUENCES.map((debug) => (
                  <button
                    key={debug.label}
                    onClick={() => resolveSequence(debug.sequence, "debug")}
                    disabled={
                      phase === "countdown" ||
                      phase === "input" ||
                      phase === "resolving" ||
                      phase === "enemy" ||
                      phase === "clear" ||
                      phase === "gameover"
                    }
                  >
                    {debug.label}
                  </button>
                ))}
              </div>
              <div className="segment-list">
                {recognizedSegments.map((segment) => (
                  <span key={segment.segment_id}>
                    #{segment.segment_id} {segment.code} axis={segment.dominant_axis}
                  </span>
                ))}
              </div>
              <pre>{battleLog.join("\n")}</pre>
            </details>
          </aside>
        </section>
        ) : (
          <section className="start-screen panel">
            <div className="start-starfield" aria-hidden="true" />
            <img className="start-logo" src="/images/m1gp-quest-logo.png" alt="M1GP QUEST" />
            <p className="start-tagline">DRAW SIGILS — CAST MAGIC</p>
            <button className="start-button" onClick={startGame}>
              <span className="press-start-text">▶ PRESS START</span>
            </button>
            <p className="start-copyright">Yusei Hida</p>
            {debugOpen ? (
              <section className="start-debug-panel" aria-label="Start screen debug settings">
                <strong>Debug</strong>
                <label className="debug-toggle-row">
                  <input type="checkbox" checked={effectsEnabled} onChange={(event) => setEffectsEnabled(event.target.checked)} />
                  Effects enabled
                </label>
                <label className="debug-toggle-row">
                  <input type="checkbox" checked={soundEffectsEnabled} onChange={(event) => setSoundEffectsEnabled(event.target.checked)} />
                  Sound effects
                </label>
                <label className="debug-toggle-row">
                  <input type="checkbox" checked={bgmEnabled} onChange={(event) => toggleBgm(event.target.checked)} />
                  BGM
                </label>
                <div className="debug-action-group">
                  <span>Tutorial</span>
                  <button className="tutorial-start-button" onClick={startTutorial}>
                    Start Tutorial
                  </button>
                </div>
                <div className="debug-action-group">
                  <span>Cast control</span>
                  <div className="debug-mode-switch" role="group" aria-label="Cast control mode">
                    <button
                      className={castControlMode === "trigger" ? "active" : ""}
                      onClick={() => void changeCastControlMode("trigger")}
                    >
                      Trigger Gesture
                    </button>
                    <button
                      className={castControlMode === "tap" ? "active" : ""}
                      onClick={() => void changeCastControlMode("tap")}
                    >
                      Tap Wand
                    </button>
                  </div>
                </div>
              </section>
            ) : null}
          </section>
        )
      ) : mode === "training" ? (
        <section className="training-layout">
          <section className="panel training-panel">
            <div>
              <p className="eyebrow">Training Mode</p>
              <h1>図形モーション学習</h1>
              <p className="hint">
                スマホで Start Streaming したまま、ラベルを選びます。Recordで記録開始、図形を描いて、Stopで保存します。
              </p>
            </div>

            <div className="field">
              <label htmlFor="training-session">Battle session</label>
              <input
                id="training-session"
                value={battleSessionId}
                onChange={(event) => setBattleSessionId(event.target.value)}
              />
              <p className="hint">スマホ側の Battle session と同じ値にしてください。初期値は battle-room です。</p>
            </div>

            <div className="training-status-grid">
              <div>
                <span>connection</span>
                <strong className={`state-${connection}`}>{connection}</strong>
              </div>
              <div>
                <span>server samples</span>
                <strong>{serverSampleCount}</strong>
              </div>
              {TRAINING_LABELS.map((label) => (
                <div key={label}>
                  <span>{trainingLabelText(label)}</span>
                  <strong>{trainingStatus.counts[label]} files</strong>
                </div>
              ))}
              <div>
                <span>model</span>
                <strong>{trainingStatus.modelExists ? "exists" : "none"}</strong>
              </div>
              <div>
                <span>model status</span>
                <strong>{trainingStatus.modelStatus}</strong>
              </div>
              <div>
                <span>trigger model</span>
                <strong>{trainingStatus.triggerModelStatus ?? "unknown"}</strong>
              </div>
            </div>

            <div className="training-record-grid">
              {TRAINING_LABELS.map((label) => (
                <button
                  key={label}
                  className=""
                  onClick={() => void recordTrainingSample(label)}
                  disabled={trainingBusy}
                >
                  Record {trainingLabelText(label)}
                </button>
              ))}
            </div>
            <div className="training-actions">
              <button onClick={() => void stopTrainingSample()} disabled={!trainingActiveLabel}>
                Stop Shape Recording
              </button>
              <button onClick={() => void refreshTrainingStatus()} disabled={trainingBusy && !trainingActiveLabel}>
                Refresh Status
              </button>
            </div>

            <div className="training-status-grid">
              <div>
                <span>Trigger</span>
                <strong>{trainingStatus.triggerCounts.trigger} files</strong>
              </div>
              <div>
                <span>Trigger none</span>
                <strong>{trainingStatus.triggerCounts.none} files</strong>
              </div>
            </div>

            <div className="training-record-grid">
              {TRIGGER_LABELS.map((label) => (
                <button key={label} onClick={() => void recordTriggerSample(label)} disabled={trainingBusy}>
                  Record Trigger {label}
                </button>
              ))}
              <button onClick={() => void stopTriggerSample()} disabled={!triggerActiveLabel}>
                Stop Trigger Recording
              </button>
            </div>

            <div className="training-actions">
              <button onClick={() => void trainMotionModel()} disabled={trainingBusy}>
                Train Shape Model
              </button>
              <button onClick={() => void trainTriggerModel()} disabled={trainingBusy}>
                Train Trigger Model
              </button>
              <button onClick={() => void archiveTriggerTraining()} disabled={trainingBusy}>
                Archive Trigger Data
              </button>
              <button onClick={() => void refreshTrainingStatus()} disabled={trainingBusy}>
                Refresh Status
              </button>
            </div>

            <div className="training-message">
              <strong>
                {trainingActiveLabel
                  ? `${trainingLabelText(trainingActiveLabel)} recording. Press Stop Shape Recording.`
                  : triggerActiveLabel
                    ? `Trigger ${triggerActiveLabel} recording. Press Stop Trigger Recording.`
                    : trainingMessage}
              </strong>
            </div>
          </section>

          <section className="panel training-guide">
            <h2>記録ガイド</h2>
            <div>
              <strong>○</strong>
              <p>Spiral: 小さく始めて外へ広げる渦巻を描く。</p>
            </div>
            <div>
              <strong>★</strong>
              <p>Star: 星形を描く。</p>
            </div>
            <div>
              <strong>△</strong>
              <p>Triangle: 三角形を描く。</p>
            </div>
            <div>
              <strong>□</strong>
              <p>Square: 四角形を描く。</p>
            </div>
            <div>
              <strong>Z</strong>
              <p>Zigzag: 左右にジグザグに動かす。</p>
            </div>
            <div>
              <strong>none</strong>
              <p>何もしない、トリガージェスチャーだけ、適当な動きなどを記録する。</p>
            </div>
            <p className="warning">none も必ず記録してください。誤認識を減らすために重要です。</p>
            <p className="hint">
              目安は各ラベル10回です。データが少ない場合でもTrain Modelは落ちにくいようにしています。
            </p>
            <p className="hint">shape model path: {trainingStatus.modelPath || "backend/models/shape_classifier.pkl"}</p>
            <p className="hint">trigger model path: {trainingStatus.triggerModelPath || "backend/models/trigger_detector.pkl"}</p>
          </section>
        </section>
      ) : mode === "tutorial" ? (
        tutorialComplete ? (
          <section className="start-screen panel">
            <img className="start-logo" src="/images/m1gp-quest-logo.png" alt="M1GP QUEST" />
            <div className="tutorial-complete-card">
              <strong>Tutorial Clear</strong>
              <span>The player has practiced the cast trigger and all three motions.</span>
            </div>
            <button className="start-button" onClick={startGame}>
              Game Start
            </button>
            <button className="tutorial-start-button" onClick={resetToStart}>
              Title
            </button>
          </section>
        ) : (
        <section className="tutorial-battle-layout">
          <aside className="panel stage-panel tutorial-steps-panel">
            <p className="eyebrow">Tutorial</p>
            <h2>M1GP Quest</h2>
            <div className="stage-current">How to Play</div>
            <div className="enemy-chain">
              {TUTORIAL_STEPS.map((step, index) => (
                <span key={step.shortTitle} className={index === tutorialStepIndex ? "active" : index < tutorialStepIndex ? "cleared" : ""}>
                  {index + 1}. {step.shortTitle}
                </span>
              ))}
            </div>
            <div className="stage-skill-list">
              <div>
                <span>Spiral</span>
                <strong>{elementLabel("water")}</strong>
              </div>
              <div>
                <span>Star</span>
                <strong>{elementLabel("fire")}</strong>
              </div>
              <div>
                <span>Z Shape</span>
                <strong>{elementLabel("thunder")}</strong>
              </div>
            </div>
          </aside>

          <div className="tutorial-battle-main">
            <section className="enemy-stage panel">
              <div className="enemy-art tutorial-art" style={stageStyle}>
                <div className="tutorial-guide-card tutorial-trigger-card">
                  <strong>{tutorialStep.kind === "trigger" ? tutorialStep.title : "Cast Trigger"}</strong>
                  <span>Push the phone forward, pause briefly, then return.</span>
                  <small>{tutorialTriggerOk ? "OK. The current trigger model detected it." : "You cannot continue until the trigger is detected."}</small>
                  {tutorialStep.kind === "trigger" ? (
                    tutorialTriggerOk ? (
                      <>
                        <div className="tutorial-ok-badge">OK</div>
                        <button onClick={() => void advanceTutorialStep()}>Next</button>
                      </>
                    ) : (
                      <button onClick={() => void startTutorialTriggerCheck()} disabled={tutorialTriggerChecking}>
                        {tutorialTriggerChecking ? "Checking..." : "Start Motion"}
                      </button>
                    )
                  ) : (
                    <div className="tutorial-ok-badge muted">Done</div>
                  )}
                </div>

                <div className="enemy-placeholder enemy-0 has-image tutorial-enemy">
                  <img
                    className="enemy-image"
                    src={enemy.image}
                    alt={enemy.name}
                    onError={(event) => event.currentTarget.classList.add("is-missing")}
                  />
                  <span>{enemy.visual}</span>
                </div>

                <div className="tutorial-guide-card tutorial-shape-card">
                  <strong>{tutorialStep.title}</strong>
                  <span>{tutorialStep.prompt}</span>
                  <small>{tutorialStep.detail}</small>
                  <div className="tutorial-shape-buttons">
                    {tutorialStep.kind === "shape" && !trainingActiveLabel ? (
                      <button onClick={() => void recordTrainingSample(tutorialStep.label, "tutorial")} disabled={trainingBusy}>
                        Start Motion
                      </button>
                    ) : null}
                    {tutorialStep.kind === "shape" && trainingActiveLabel ? (
                      <>
                        <button onClick={() => void stopTrainingSample(tutorialStep.label)}>
                          Save
                        </button>
                        <button onClick={() => void discardTrainingSample()}>
                          Discard
                        </button>
                      </>
                    ) : null}
                  </div>
                </div>

                <div className="enemy-summary enemy-stage-card tutorial-status-card">
                  <div>
                    <strong>{tutorialStep.title}</strong>
                  </div>
                  <div>
                    <span>Player action</span>
                    <strong>{trainingActiveLabel ? "Now recording motion..." : tutorialStep.prompt}</strong>
                    <small>{tutorialStep.kind === "shape" ? "Developer chooses Save or Discard after the action." : "Tutorial waits for the current trigger model to detect this gesture."}</small>
                  </div>
                </div>

                <div className="enemy-speech-card tutorial-message-card">
                  <span className="enemy-speech-text">
                    {trainingActiveLabel
                      ? "Keep moving. The developer will save this attempt."
                      : tutorialStep.kind === "trigger"
                        ? "First, learn the cast trigger gesture."
                        : "Now try this motion with the phone."}
                  </span>
                </div>
              </div>
            </section>

            <section className="input-banner tutorial-bottom-bar">
              <div className="input-banner-grid">
                <div className="attack-result-card">
                  <strong>Current Tutorial</strong>
                  <small className="english-result">
                    {tutorialStep.detail}
                  </small>
                </div>
                <div className="player-status-card">
                  <strong>Developer</strong>
                  <small>{trainingActiveLabel ? "Choose Save or Discard." : "The player only moves the phone; the developer controls these buttons."}</small>
                </div>
              </div>
            </section>
          </div>
        </section>
        )
      ) : false ? (
        <section className="training-layout tutorial-layout">
          <section className="panel training-panel">
            <div>
              <p className="eyebrow">Tutorial Mode</p>
              <h1>図形入力チュートリアル</h1>
              <p className="hint">
                手順は共通です。Recordで記録開始、平面に図形を描き、Stopで保存します。
                保存データには開始/終了トリガーを含めません。
              </p>
            </div>
            <div className="tutorial-grid">
              {TRAINING_LABELS.map((label) => (
                <div className="tutorial-card" key={label}>
                  <strong>{trainingLabelText(label)}</strong>
                  <p>
                    {label === "circle"
                      ? "小さく始めて外へ広げる渦巻を描きます。円より区別しやすい動きです。"
                      : label === "star"
                        ? "星形を描くつもりで上下左右に大きく動かします。"
                        : label === "triangle"
                          ? "三角形の3辺をなぞるように動かします。"
                          : label === "square"
                            ? "四角形の4辺をなぞるように動かします。"
                            : label === "zigzag"
                              ? "左右に折り返しながらジグザグに動かします。"
                              : "何もしない、トリガーだけ、適当な動きを記録します。"}
                  </p>
                  <small>{trainingStatus.counts[label]} files</small>
                  <button onClick={() => void recordTrainingSample(label, "tutorial")} disabled={trainingBusy}>
                    Record
                  </button>
                </div>
              ))}
            </div>
            <div className="training-actions">
              <button onClick={() => void stopTrainingSample()} disabled={!trainingActiveLabel}>
                Stop
              </button>
              <button onClick={() => void trainMotionModel()} disabled={trainingBusy && !trainingActiveLabel}>
                Train Shape Model
              </button>
            </div>
          </section>
          <section className="panel training-guide">
            <h2>現在の状態</h2>
            <div>
              <strong>{trainingActiveLabel ? trainingLabelText(trainingActiveLabel as TrainingLabel) : "-"}</strong>
              <p>{trainingMessage}</p>
            </div>
            <p className="hint">スマホ側は Phone 画面で Start Streaming を押したままにしてください。</p>
            <p className="hint">目安は各図形10回、最初の動作確認は各5回でも十分です。</p>
          </section>
        </section>
      ) : phoneView === "play" ? (
        <section className={`phone-play-screen phone-state-${phoneMotionState} phone-pulse-${phoneCastPulse || "idle"}`}>
          <button className="phone-ui-button" onClick={() => setPhoneView("controller")} aria-label="Open phone controls">
            UI
          </button>
          <button
            className={`phone-wand-button ${castControlMode === "tap" ? "is-tappable" : ""}`}
            onClick={() => void handlePhoneWandTap()}
            disabled={castControlMode !== "tap" || !isSensorStreaming || phoneTapBusy}
            aria-label={castControlMode === "tap" ? "Start or finish casting" : "Casting wand"}
          >
            <span className="phone-wand-aura" aria-hidden="true">
              <img className="phone-wand" src="/images/casting-wand.png" alt="" />
            </span>
          </button>
        </section>
      ) : (
        <section className="phone-layout">
          <section className="panel">
            <div className="phone-controller-heading">
              <h2>Phone Sensor Controller</h2>
              <button className="phone-play-button" onClick={() => setPhoneView("play")}>Play Screen</button>
            </div>
            <div className="field">
              <label htmlFor="sensor-session">Battle session</label>
              <input
                id="sensor-session"
                value={sensorSessionId}
                onChange={(event) => setSensorSessionId(event.target.value)}
              />
              <p className="hint">Use the same session as the PC battle screen.</p>
            </div>
            <div className="field">
              <label htmlFor="api-base-url">API base URL</label>
              <input
                id="api-base-url"
                value={apiBaseUrl}
                onChange={(event) => setApiBaseUrl(event.target.value)}
                placeholder="blank = same origin proxy (/api)"
                inputMode="url"
              />
              <p className={isMixedContentRisk ? "warning" : "hint"}>
                POST endpoint: {endpoint}
                {isMixedContentRisk ? " HTTPS page to HTTP API may be blocked by Safari." : ""}
              </p>
            </div>

            <div className="button-row">
              <button onClick={() => void requestPermission()}>Allow Sensor</button>
              <button onClick={() => void startSensorStream()} disabled={isSensorStreaming}>
                Start Streaming
              </button>
              <button onClick={() => void stopSensorStream()} disabled={!isSensorStreaming}>
                Stop
              </button>
            </div>

            <div className="status-grid">
              <div>
                <span>permission</span>
                <strong>{permission}</strong>
              </div>
              <div>
                <span>secure context</span>
                <strong>{isSecurePage ? "yes" : "no"}</strong>
              </div>
              <div>
                <span>connection</span>
                <strong className={`state-${connection}`}>{connection}</strong>
              </div>
              <div>
                <span>sampling count</span>
                <strong>{sampleCount}</strong>
              </div>
              <div>
                <span>sent count</span>
                <strong>{sentCount}</strong>
              </div>
              <div>
                <span>queued samples</span>
                <strong>{queuedSamples}</strong>
              </div>
              <div>
                <span>last send status</span>
                <strong>{lastSendStatus}</strong>
              </div>
              <div>
                <span>last send error</span>
                <strong>{lastSendError || "-"}</strong>
              </div>
              <div>
                <span>send timer active</span>
                <strong>{sendTimerActive ? "yes" : "no"}</strong>
              </div>
              <div>
                <span>last flush sample count</span>
                <strong>{lastFlushSampleCount}</strong>
              </div>
              <div>
                <span>last flush time</span>
                <strong>{lastFlushTime}</strong>
              </div>
            </div>
            <p className="hint">permission detail: {permissionDetail}</p>
          </section>

          <section className="readout">
            <div>
              <span>x</span>
              <strong>{formatValue(current.x)}</strong>
            </div>
            <div>
              <span>y</span>
              <strong>{formatValue(current.y)}</strong>
            </div>
            <div>
              <span>z</span>
              <strong>{formatValue(current.z)}</strong>
            </div>
            <div>
              <span>magnitude</span>
              <strong>{formatValue(current.magnitude)}</strong>
            </div>
          </section>

          <section className="panel compact">
            <h2>Send Log</h2>
            <pre>{sensorLogs.length > 0 ? sensorLogs.join("\n") : "No logs yet."}</pre>
          </section>
        </section>
      )}
    </main>
  );
}

export default App;

export type EnemyDisplayConfig = {
  displayName: string;
  images: {
    normal: string;
    defeated: string;
    stage2?: string;
    stage3?: string;
  };
  dialogue: {
    normal: string;
    defeated: string;
  };
};

export const ENEMY_DISPLAY: Record<string, EnemyDisplayConfig> = {
  enemy1: {
    displayName: "enemy1",
    images: {
      normal: "/images/enemy1.png",
      defeated: "/images/enemy1.png",
    },
    dialogue: {
      normal: "セリフ1",
      defeated: "セリフ2",
    },
  },
  enemy2: {
    displayName: "enemy2",
    images: {
      normal: "/images/enemy2.png",
      defeated: "/images/enemy2.png",
    },
    dialogue: {
      normal: "セリフ1",
      defeated: "セリフ2",
    },
  },
  enemy3: {
    displayName: "enemy3",
    images: {
      normal: "/images/enemy3.png",
      defeated: "/images/enemy3.png",
    },
    dialogue: {
      normal: "セリフ1",
      defeated: "セリフ2",
    },
  },
  enemy4: {
    displayName: "enemy4",
    images: {
      normal: "/images/enemy4.png",
      defeated: "/images/enemy4.png",
    },
    dialogue: {
      normal: "セリフ1",
      defeated: "セリフ2",
    },
  },
};

const DEFAULT_DISPLAY: EnemyDisplayConfig = {
  displayName: "enemy",
  images: {
    normal: "/images/enemy1.png",
    defeated: "/images/enemy1.png",
  },
  dialogue: {
    normal: "...",
    defeated: "...",
  },
};

export function getEnemyDisplay(enemyName: string): EnemyDisplayConfig {
  return ENEMY_DISPLAY[enemyName] ?? DEFAULT_DISPLAY;
}

export function getEnemyImage(enemyName: string, hp: number, hitStage = 0): string {
  const config = getEnemyDisplay(enemyName);
  if (hp <= 0) {
    return config.images.defeated;
  }

  const stages = [config.images.normal, config.images.stage2, config.images.stage3].filter(
    (path): path is string => Boolean(path),
  );
  const index = Math.min(Math.max(0, hitStage), stages.length - 1);
  return stages[index] ?? config.images.normal;
}

# M1GP QUEST

**スマートフォンを魔法の杖のように振って操作する、加速度センサを利用したモーション認識RPGのプロトタイプです。**

スマートフォンから連続的に取得した加速度データの中から動作区間を切り出し、形状を認識して、Spiral・Star・Zのゲーム内スキルとして発動します。センサ入力、区間抽出、機械学習・時系列照合、ゲームでのフィードバックを一つの動作するデモにまとめた個人制作です。

> このプロジェクトはスマートホームを対象とする現在の研究そのものではありません。連続センサログの区間抽出と分類という共通する考え方を、ゲームとして分かりやすく応用したプロトタイプです。

## デモの特徴

- スマートフォンのブラウザを入力デバイスとして使用
- DeviceMotion APIから加速度データを連続取得
- 意図した動作の開始・終了区間を抽出
- DTWを優先し、Random Forestをフォールバックとして動作分類
- 認識結果をターン制RPGのスキルとして即時に可視化
- PC戦闘画面、スマートフォン入力画面、学習データ収集画面、デバッグ機能を用意

現在のメインモード `GAME2` では、次の3動作を扱います。

| Motion | Skill | 消費MP | 効果 |
|---|---|---:|---|
| Spiral | Spiral | 0 | 20ダメージ |
| Star | Star | 3 | 60ダメージ |
| Z | Z | 3 | 敵の次の攻撃をスタン |

## 処理フロー

```text
スマートフォン
    ↓ DeviceMotion API
連続加速度ストリーム
    ↓
動作区間の抽出
    ↓
トリミング・リサンプリング・正規化
    ↓
DTWテンプレート照合 / Random Forest
    ↓
動作分類（Spiral / Star / Z / none）
    ↓
ゲーム内スキル発動
```

区間の開始・終了がずれると、分類器へ不要なデータや不完全な動作が入力され、認識が不安定になります。本プロトタイプでは、区間抽出と分類を分けて扱うことで、この関係をゲーム上の成否として確認できます。

## 技術スタック

| レイヤ | 技術 |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| Backend | Python, 標準HTTPサーバ |
| Sensor | DeviceMotion API, 3軸加速度データ |
| Recognition | DTW, Random Forest, scikit-learn |
| Data Processing | NumPy, pandas |
| Visualization / Analysis | matplotlib |
| Demo Connection | cloudflared |

## システム構成

```text
Smartphone Browser
  └─ DeviceMotion API
          │ batched acceleration samples
          ▼
Python Backend
  ├─ センサデータ受信・セッション管理
  ├─ 動作区間抽出
  ├─ DTW / Random Forestによる認識
  ├─ 学習データ保存
  └─ デモ時はビルド済みFrontendを配信
          │ recognition result
          ▼
React Frontend
  ├─ PC戦闘画面
  ├─ スマートフォン入力画面
  ├─ 学習画面
  └─ チュートリアル / デバッグパネル
```

主なディレクトリ:

```text
m1gp-quest/
├─ backend/
│  ├─ server.py                 # 受信、区間抽出、認識、ゲームAPI
│  ├─ scripts/                  # 学習・分析・可視化
│  ├─ data/training/            # 形状分類用データ
│  ├─ data/training_trigger/    # トリガ検出用データ
│  └─ models/                   # 学習済みモデル
├─ frontend/
│  ├─ src/                      # React / TypeScript実装
│  ├─ public/                   # 画像・音声などのデモ素材
│  └─ dist/                     # ポータブルデモ用ビルド成果物
├─ StartDevDemo.bat / .ps1      # 開発モード
└─ StartDemo.bat / .ps1         # ビルド済みデモ
```

## モーション認識

### 1. 区間抽出

スマートフォンから届く連続加速度データに対し、トリガ動作または画面上の杖ボタンを使って、分類対象となる描画区間を決定します。

- **Trigger gesture mode:** スマートフォンを前へ突き出して戻す動作で詠唱を開始・確定
- **Tap wand mode:** 画面上の杖をタップして詠唱を開始・確定

Tap wand modeは、トリガ認識の影響を除き、形状認識だけを確認したい場合に利用できます。

### 2. DTWによる形状照合

1. 入力区間の前後をトリミング
2. 時系列を固定長へリサンプリング
3. 軸ごとに正規化
4. 保存済みテンプレートとのDTW距離を計算
5. 最も近いラベルを選択
6. 距離・信頼度の閾値で不確かな入力を `none` として棄却

対象ラベルは `circle`（Spiral）、`star`（Star）、`zigzag`（Z）、`none` です。DTWで判定できない場合はRandom Forestをフォールバックとして使用します。

### 3. 学習データの分離

形状分類用データとトリガ検出用データを別ディレクトリで管理しています。トリガ動作が形状データへ混入すると分類が不安定になるため、それぞれの対象区間だけを保存します。

## 研究との関係

現在取り組んでいるスマートホーム行動認識研究とは対象・データが異なります。一方、次の技術的な問いは共通しています。

- 連続したセンサログから、意味のある動作区間をどのように切り出すか
- 抽出した時系列区間をどのように分類するか
- 区間の切り方が認識結果へどのように影響するか

研究上の課題を説明可能な形にし、入力から結果までを体験できるデモへ落とし込むことを目的としています。

## 実行方法

### 必要環境

- Windows
- Python 3.10以上
- Node.js / npm（開発モードのみ）
- スマートフォン（DeviceMotion API対応ブラウザ）
- `cloudflared.exe`（リポジトリ直下、またはPATH上）

### 開発モード

ViteとPythonバックエンドを起動します。

```powershell
git clone https://github.com/hidayusei/m1gp-quest.git
cd m1gp-quest

python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
npm --prefix frontend install

.\StartDevDemo.bat
```

起動後のURL:

- PC画面: `http://127.0.0.1:5173/`
- Backend: `http://127.0.0.1:8000/`
- スマートフォン: 起動時に表示されるHTTPS URL

### ビルド済みデモ

リポジトリに含まれる `frontend/dist` をPythonバックエンドから配信します。

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
.\StartDemo.bat
```

PC画面は `http://127.0.0.1:8000/` で開きます。スマートフォンでは、起動時に表示されるcloudflaredのHTTPS URLへ `?mode=phone` を付けたURLを使用します。

> cloudflared経由のURLは外部からアクセス可能になります。デモ中のみ起動し、終了後はランチャーから停止してください。

## 開発用コマンド

Frontend:

```powershell
npm --prefix frontend install
npm --prefix frontend run build
```

Backend:

```powershell
backend/.venv/Scripts/python backend/server.py
```

分析・再学習用スクリプトは `backend/scripts/` にあります。

## リポジトリ公開時の方針

- `frontend/dist` はポータブルデモをすぐ起動できるよう意図的に追跡しています。
- `backend/data/training*` と `backend/models` は認識デモに必要な学習サンプル・学習済みモデルとして追跡しています。
- 実行時ログ、追加収集した生データ、仮想環境、キャッシュ、環境変数ファイルはGit管理しません。

## 現在の制約

- 認識結果は端末の持ち方、動かす速さ、区間の切り方などに影響されます。
- `backend/server.py` は複数の責務を持つ2,000行以上の大きなファイルです。今回は動作維持を優先していますが、今後はHTTP処理、セッション管理、区間抽出、分類器を段階的に分割する余地があります。
- デモ素材の利用条件・出典は、公開前に各ファイルについて確認する必要があります。

## License

現時点ではライセンスを設定していません。ソースコードや素材の再利用条件は明示していません。

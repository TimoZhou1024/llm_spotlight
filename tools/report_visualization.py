#!/usr/bin/env python3
import os
import glob
import json
import argparse
import textwrap
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Constants
# ==========================================
# The fallback text used in the codebase when a model outputs invalid JSON during debate
SKIP_PHRASE = "I choose to skip my turn and not make a statement this round."

# Plotting settings
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16


# ==========================================
# Utility Functions
# ==========================================
def wrap_text(text, width=15):
    """Wrap long model names for better visualization."""
    if not isinstance(text, str):
        return str(text)
    return textwrap.fill(text, width=width)

def save_fig(fig, out_dir, name):
    """Save figure in both PNG and SVG formats with tight layout."""
    png_path = os.path.join(out_dir, f"{name}.png")
    svg_path = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def get_latest_csv(logs_root):
    """Find the latest eval_results_*.csv in the given directory."""
    pattern = os.path.join(logs_root, "eval_results_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ==========================================
# Data Loading & Processing
# ==========================================
def parse_session_log(log_dir):
    """Parse game_complete.json to extract game stats and skips."""
    game_file = os.path.join(log_dir, "game_complete.json")
    stats = {
        "Rounds": 0,
        "VillagerDebateTurns": 0,
        "WerewolfDebateTurns": 0,
        "VillagerSkips": 0,
        "WerewolfSkips": 0
    }
    if not os.path.exists(game_file):
        return stats
    
    try:
        with open(game_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        stats["Rounds"] = len(data.get("rounds", []))
        
        # Try to identify werewolf player names to distinguish roles during debate
        werewolves_list = data.get("werewolves", [])
        # Sometimes it's a list of dicts, sometimes strings
        ww_names = set()
        for ww in werewolves_list:
            if isinstance(ww, dict):
                ww_names.add(ww.get("name", ""))
            else:
                ww_names.add(str(ww))
        
        # Scan debates for fallback / skip phrase
        # Note: We group active skips and JSON fallbacks together here as they output the same phrase.
        for r in data.get("rounds", []):
            debate = r.get("debate", {})
            if isinstance(debate, dict):
                for player_name, statement in debate.items():
                    is_ww = player_name in ww_names
                    is_skip = SKIP_PHRASE in str(statement)
                    
                    if is_ww:
                        stats["WerewolfDebateTurns"] += 1
                        if is_skip: stats["WerewolfSkips"] += 1
                    else:
                        stats["VillagerDebateTurns"] += 1
                        if is_skip: stats["VillagerSkips"] += 1
    except Exception as e:
        print(f"[Warning] Failed to parse {game_file}: {e}")
        
    return stats

def load_data(csv_path, logs_root):
    """Load evaluation CSV and enrich it with session logs."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("[Warning] Eval CSV not found. Attempting to scan logs directory...")
        session_dirs = glob.glob(os.path.join(logs_root, "session_*"))
        if not session_dirs:
            raise FileNotFoundError("No eval CSV and no session logs found.")
        
        # Mock dataframe from session dirs
        rows = []
        for sdir in session_dirs:
            gf = os.path.join(sdir, "game_complete.json")
            if os.path.exists(gf):
                try:
                    with open(gf, 'r') as f:
                        d = json.load(f)
                        winner = d.get('winner', 'Unknown')
                        rows.append({
                            'VillagerModel': 'Unknown',
                            'WerewolfModel': 'Unknown',
                            'Winner': winner,
                            'WinnerModel': 'Unknown',
                            'Log': sdir
                        })
                except:
                    pass
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Failed to extract any data from logs directory.")
        print("[Notice] Using fallback scanning. Model assignments may be unknown.")

    # Ensure required columns exist
    for col in ['VillagerModel', 'WerewolfModel', 'Winner', 'Log']:
        if col not in df.columns:
            df[col] = "Unknown"
            
    if 'WinnerModel' not in df.columns:
        df['WinnerModel'] = None

    enhanced_records = []
    for idx, row in df.iterrows():
        # Handle missing WinnerModel
        winner_model = row['WinnerModel']
        if pd.isna(winner_model) or winner_model == "" or winner_model == "Unknown":
            if row['Winner'] == 'Villagers':
                winner_model = row['VillagerModel']
            elif row['Winner'] == 'Werewolves':
                winner_model = row['WerewolfModel']
            else:
                winner_model = "Draw/Error"
        
        # Default Log to relative if log directory passed (handling cross-platform paths)
        log_path = row['Log']
        if not os.path.exists(log_path) and logs_root:
            alt_path = os.path.join(logs_root, os.path.basename(log_path))
            if os.path.exists(alt_path):
                log_path = alt_path

        stats = parse_session_log(log_path)
        
        enhanced_records.append({
            'GameIndex': idx + 1,
            'VillagerModel': row['VillagerModel'],
            'WerewolfModel': row['WerewolfModel'],
            'WinnerFaction': row['Winner'],
            'WinnerModel': winner_model,
            'Rounds': stats['Rounds'],
            'VillagerDebateTurns': stats['VillagerDebateTurns'],
            'WerewolfDebateTurns': stats['WerewolfDebateTurns'],
            'VillagerSkips': stats['VillagerSkips'],
            'WerewolfSkips': stats['WerewolfSkips'],
            'LogPath': log_path,
            'Matchup': f"{wrap_text(row['VillagerModel'], 10)} (V) vs {wrap_text(row['WerewolfModel'], 10)} (W)"
        })

    return pd.DataFrame(enhanced_records)

def generate_summaries(df):
    """Generate model and matchup summary tables."""
    # 1. Model Summary
    models = set(df['VillagerModel'].unique()).union(set(df['WerewolfModel'].unique()))
    models.discard("Unknown")
    models.discard("Draw/Error")
    
    model_stats = []
    for m in models:
        # As Villager
        v_df = df[df['VillagerModel'] == m]
        v_games = len(v_df)
        v_wins = len(v_df[v_df['WinnerFaction'] == 'Villagers'])
        v_turns = v_df['VillagerDebateTurns'].sum()
        v_skips = v_df['VillagerSkips'].sum()
        
        # As Werewolf
        w_df = df[df['WerewolfModel'] == m]
        w_games = len(w_df)
        w_wins = len(w_df[w_df['WinnerFaction'] == 'Werewolves'])
        w_turns = w_df['WerewolfDebateTurns'].sum()
        w_skips = w_df['WerewolfSkips'].sum()
        
        total_games = v_games + w_games
        total_wins = v_wins + w_wins
        total_turns = v_turns + w_turns
        total_skips = v_skips + w_skips
        
        # Avg Rounds where the model was involved
        avg_rounds = df[(df['VillagerModel'] == m) | (df['WerewolfModel'] == m)]['Rounds'].mean()
        
        model_stats.append({
            'Model': m,
            'Games': total_games,
            'Wins': total_wins,
            'WinRate': total_wins / total_games if total_games > 0 else 0,
            'VillagerGames': v_games,
            'VillagerWins': v_wins,
            'VillagerWinRate': v_wins / v_games if v_games > 0 else 0,
            'WerewolfGames': w_games,
            'WerewolfWins': w_wins,
            'WerewolfWinRate': w_wins / w_games if w_games > 0 else 0,
            'AvgRounds': avg_rounds,
            'DebateTurns': total_turns,
            'SkipCount': total_skips,
            'SkipRate': total_skips / total_turns if total_turns > 0 else 0
        })
    df_model = pd.DataFrame(model_stats)
    if not df_model.empty:
        df_model.sort_values(by='WinRate', ascending=False, inplace=True)
    
    # 2. Matchup Summary
    matchup_stats = []
    for (v_mod, w_mod), group in df.groupby(['VillagerModel', 'WerewolfModel']):
        games = len(group)
        v_wins = len(group[group['WinnerFaction'] == 'Villagers'])
        w_wins = len(group[group['WinnerFaction'] == 'Werewolves'])
        
        top_winner_model = v_mod if v_wins >= w_wins else w_mod
        top_winner_rate = max(v_wins, w_wins) / games if games > 0 else 0
        
        matchup_stats.append({
            'VillagerModel': v_mod,
            'WerewolfModel': w_mod,
            'Games': games,
            'VillagerSideWins': v_wins,
            'WerewolfSideWins': w_wins,
            'VillagerSideWinRate': v_wins / games if games > 0 else 0,
            'TopWinnerModel': top_winner_model,
            'TopWinnerRate': top_winner_rate,
            'AvgRounds': group['Rounds'].mean(),
            'SkipCount': group['VillagerSkips'].sum() + group['WerewolfSkips'].sum()
        })
    df_matchup = pd.DataFrame(matchup_stats)
    
    return df_model, df_matchup

# ==========================================
# Plotting Functions
# ==========================================
def plot_overall_win_rate(df_model, out_dir):
    if df_model.empty: return
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_model) * 0.8)))
    df_model['Model_Wrapped'] = df_model['Model'].apply(lambda x: wrap_text(x, 20))
    
    bars = sns.barplot(
        x='WinRate', y='Model_Wrapped', 
        data=df_model, ax=ax, 
        color=sns.color_palette("colorblind")[0],
        edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Overall Win Rate")
    ax.set_ylabel("Model")
    ax.set_title("Overall Win Rate by Model", pad=20)
    
    # Annotations
    for idx, row in enumerate(df_model.itertuples()):
        text = f" {row.Wins}/{row.Games} ({row.WinRate:.1%})"
        ax.text(row.WinRate, idx, text, va='center', ha='left', fontsize=10)
        
    save_fig(fig, out_dir, "A_overall_win_rate")

def plot_role_specific_win_rate(df_model, out_dir):
    if df_model.empty: return
    
    melted = []
    for _, row in df_model.iterrows():
        melted.append({'Model': row['Model'], 'Role': 'As Villager (Faction)', 'WinRate': row['VillagerWinRate'], 'Str': f"{row['VillagerWins']}/{row['VillagerGames']}"})
        melted.append({'Model': row['Model'], 'Role': 'As Werewolf (Faction)', 'WinRate': row['WerewolfWinRate'], 'Str': f"{row['WerewolfWins']}/{row['WerewolfGames']}"})
    
    df_melt = pd.DataFrame(melted)
    df_melt['Model_Wrapped'] = df_melt['Model'].apply(lambda x: wrap_text(x, 15))
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_model) * 1.2)))
    bars = sns.barplot(
        x='WinRate', y='Model_Wrapped', hue='Role', 
        data=df_melt, ax=ax, edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Model")
    ax.set_title("Win Rate by Controlled Faction", pad=20)
    ax.legend(title="Controlled Role", loc='lower right')
    
    # Annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=5, fontsize=9)
        
    save_fig(fig, out_dir, "B_role_specific_win_rate")

def plot_head_to_head_heatmap(df_matchup, out_dir):
    if df_matchup.empty: return
    
    pivot_vp = df_matchup.pivot(index="VillagerModel", columns="WerewolfModel", values="VillagerSideWinRate")
    
    # Wrap labels
    pivot_vp.index = [wrap_text(idx, 15) for idx in pivot_vp.index]
    pivot_vp.columns = [wrap_text(col, 15) for col in pivot_vp.columns]
    
    fig, ax = plt.subplots(figsize=(max(6, len(pivot_vp.columns)*1.5), max(5, len(pivot_vp.index)*1.2)))
    
    sns.heatmap(
        pivot_vp, annot=True, fmt=".1%", cmap="RdBu", center=0.5,
        cbar_kws={'label': 'Villager-side Win Rate'},
        linewidths=1, linecolor='white', ax=ax,
        vmin=0, vmax=1
    )
    
    ax.set_title("Head-to-Head: Villager Model Win Rate", pad=20)
    ax.set_xlabel("Werewolf Model (Opponent)")
    ax.set_ylabel("Villager Model")
    
    save_fig(fig, out_dir, "C_head_to_head_heatmap")

def plot_game_timeline(df_games, out_dir):
    if df_games.empty: return
    
    fig, ax = plt.subplots(figsize=(12, max(5, len(df_games['Matchup'].unique())*1.5)))
    
    sns.scatterplot(
        x='GameIndex', y='Matchup', 
        hue='WinnerModel', style='WinnerFaction', size='Rounds',
        sizes=(50, 300), alpha=0.8, edgecolor='black',
        data=df_games, ax=ax
    )
    
    ax.set_title("Game Outcomes Timeline", pad=20)
    ax.set_xlabel("Game Index (Chronological)")
    ax.set_ylabel("Matchup: Villager Model vs Werewolf Model")
    
    # Fix legend outside plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    save_fig(fig, out_dir, "D_game_timeline")

def plot_rounds_distribution(df_games, out_dir):
    if df_games.empty: return
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_games['Matchup'].unique())*1.2)))
    
    sns.boxplot(
        x='Rounds', y='Matchup', 
        data=df_games, ax=ax,
        color=sns.color_palette("colorblind")[2],
        width=0.6, fliersize=5
    )
    
    ax.set_title("Distribution of Game Rounds per Matchup", pad=20)
    ax.set_xlabel("Number of Rounds")
    ax.set_ylabel("Matchup")
    
    save_fig(fig, out_dir, "E_rounds_distribution")

def plot_fallback_or_skip_rate(df_model, out_dir):
    if df_model.empty: return
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_model) * 0.8)))
    df_model['Model_Wrapped'] = df_model['Model'].apply(lambda x: wrap_text(x, 20))
    
    # Fallback/Skip rate overall
    sns.barplot(
        x='SkipRate', y='Model_Wrapped', 
        data=df_model, ax=ax,
        color=sns.color_palette("colorblind")[3],
        edgecolor='black', linewidth=1.2
    )
    
    ax.set_xlabel("Skip/Fallback Rate (per debate turn)")
    ax.set_ylabel("Model")
    ax.set_title("Debate Turn Skip or Invalid JSON Fallback Rate", pad=20)
    
    for idx, row in enumerate(df_model.itertuples()):
        text = f" {row.SkipCount}/{row.DebateTurns} ({row.SkipRate:.1%})"
        ax.text(row.SkipRate, idx, text, va='center', ha='left', fontsize=10)
        
    save_fig(fig, out_dir, "F_fallback_or_skip_rate")

# ==========================================
# Output Generation
# ==========================================
def export_tables(df_games, df_model, df_matchup, out_dir):
    # Model Summary
    df_model.to_csv(os.path.join(out_dir, "1_model_summary.csv"), index=False)
    with open(os.path.join(out_dir, "1_model_summary.md"), 'w') as f:
        f.write("### Model Summary\n\n")
        f.write(df_model.to_markdown(index=False, floatfmt=".1%"))
        
    # Matchup Summary
    df_matchup.to_csv(os.path.join(out_dir, "2_matchup_summary.csv"), index=False)
    with open(os.path.join(out_dir, "2_matchup_summary.md"), 'w') as f:
        f.write("### Matchup Summary\n\n")
        f.write(df_matchup.to_markdown(index=False, floatfmt=".1%"))
        
    # Game Level
    clean_cols = ['GameIndex', 'VillagerModel', 'WerewolfModel', 'WinnerFaction', 'WinnerModel', 'Rounds', 'VillagerSkips', 'WerewolfSkips', 'LogPath']
    df_games[clean_cols].to_csv(os.path.join(out_dir, "3_game_level_results.csv"), index=False)

def generate_readme(out_dir):
    content = """# Werewolf Benchmark Report Assets

This directory contains finalized assets (plots and tables) summarizing the evaluation of LLMs in the Werewolf multi-agent benchmark. 

## Charts
- **A_overall_win_rate**: Horizontal bar chart showing the total win rate of each model regardless of role.
- **B_role_specific_win_rate**: Grouped bar chart breaking down win rates when controlling Villagers vs Werewolves.
- **C_head_to_head_heatmap**: Win rate matrix showing how models perform against each other (displayed from the Villager model's perspective).
- **D_game_timeline**: Chronological scatter plot of game outcomes, highlighting winner models, factions, and game length (rounds).
- **E_rounds_distribution**: Boxplots showing the variance in game lengths across different matchups.
- **F_fallback_or_skip_rate**: Analysis of debate robustness, showing how often a model skipped a turn (including fallbacks caused by Invalid JSON generation).

## Summary Tables
- **1_model_summary**: Aggregate performance metrics per model. (.csv / .md)
- **2_matchup_summary**: Win rates and metrics broken down by specific model matchups. (.csv / .md)
- **3_game_level_results**: Raw augmented dataset containing every game's stats. (.csv)

*Note: All plots are provided in both PNG format (for direct viewing/presentations) and SVG format (vector graphics for academic papers).*
"""
    with open(os.path.join(out_dir, "README.md"), 'w') as f:
        f.write(content)

# ==========================================
# Main CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready visualizations for Werewolf Benchmark.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a specific eval_results CSV.")
    parser.add_argument("--logs-root", type=str, default="logs", help="Directory containing eval CSVs and session logs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for report assets.")
    
    args = parser.parse_args()
    
    # Resolve CSV and Output Dir
    csv_target = args.csv if args.csv else get_latest_csv(args.logs_root)
    
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%md_%H%M%S")
        out_dir = os.path.join(args.logs_root, f"report_assets_{timestamp}")
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Report generation started.")
    print(f"[*] Reading eval data from: {csv_target or 'Fallback to Log scanning'}")
    print(f"[*] Output directory: {out_dir}")
    
    # 1. Load Data
    try:
        df_games = load_data(csv_target, args.logs_root)
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return

    print(f"[*] Successfully loaded {len(df_games)} game records.")
    
    # 2. Compute Summaries
    df_model, df_matchup = generate_summaries(df_games)
    
    # 3. Generate Tables
    export_tables(df_games, df_model, df_matchup, out_dir)
    print(f"[*] Generated Markdown/CSV Summary tables.")

    # 4. Generate Plots
    plot_overall_win_rate(df_model, out_dir)
    plot_role_specific_win_rate(df_model, out_dir)
    plot_head_to_head_heatmap(df_matchup, out_dir)
    plot_game_timeline(df_games, out_dir)
    plot_rounds_distribution(df_games, out_dir)
    plot_fallback_or_skip_rate(df_model, out_dir)
    print(f"[*] Generated A-F Visualizations (PNG & SVG).")
    
    # 5. README
    generate_readme(out_dir)
    
    print(f"[Success] All report assets have been exported to: {out_dir}")

if __name__ == "__main__":
    main()
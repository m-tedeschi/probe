from nba_api.stats.endpoints import leaguedashplayerstats

season = "2025-26"

df = leaguedashplayerstats.LeagueDashPlayerStats(
    season=season,
    season_type_all_star="Regular Season",
    per_mode_detailed="Totals"
).get_data_frames()[0]

cols = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GP",
    "PTS",
    "AST",
    "BLK",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FG_PCT",
]

sample = df[cols].head(10)

sample.to_csv("nba_player_stats_sample.csv", index=False)

print(sample)
print("Wrote nba_player_stats_sample.csv")

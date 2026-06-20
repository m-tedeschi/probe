import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

SEASON = "2025-26"
PLAYER_NAME = "Jayson Tatum"
OUTFILE = "jayson_tatum_game_logs_2025_26.csv"


def main():
    matches = players.find_players_by_full_name(PLAYER_NAME)

    if not matches:
        raise RuntimeError(f"Could not find player: {PLAYER_NAME}")

    player_id = matches[0]["id"]

    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=SEASON,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    cols = [
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
    ]

    df = df[cols]

    for _, row in df.iterrows():
        print(
            f"{row['GAME_DATE']} | {row['MATCHUP']} | "
            f"PTS={row['PTS']} AST={row['AST']} BLK={row['BLK']} "
            f"FG={row['FGM']}/{row['FGA']} 3PT={row['FG3M']}/{row['FG3A']} "
            f"MIN={row['MIN']}"
        )

    df.to_csv(OUTFILE, index=False)
    print(f"\nWrote {len(df)} games to {OUTFILE}")


if __name__ == "__main__":
    main()

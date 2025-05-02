import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime, timedelta
import math
import torch.nn as nn
import requests as r
from bs4 import BeautifulSoup as bs
import umap


class NFLUtils():
    """
    -- Backtest functions --
    kelly_criterion-> Used by backtest_model to calculate optimal position size
    map_losses     -> Used by optuna in NFL_ANN to map b/t loss functions
    backtest_model -> Calculates and charts a backtest against the performance set
    
    -- Sliding window functions ---
    get_track_dict -> creates the track dict which is to be used by 
    """
    team_abbrv = {
        "Buffalo Bills": "buf",
        "Los Angeles Rams": "lar", # St. Louis Rams 1995-2015
        "New Orleans Saints": "nor",
        "Atlanta Falcons": "atl",
        "Cleveland Browns": "cle",
        "Carolina Panthers": "car",
        "Chicago Bears": "chi",
        "San Francisco 49ers": "sfo",
        "Pittsburgh Steelers": "pit",
        "Cincinnati Bengals": "cin",
        "Houston Texans": "hou", # Didn't exist until 2002
        "Indianapolis Colts": "ind",
        "Philadelphia Eagles": "phi",
        "Detroit Lions": "det",
        "Washington Commanders": "was", 
        "Washington Football Team": "was",
        "Washington Redskins": "was", # Washington Redskins
        "Jacksonville Jaguars": "jax",
        "Miami Dolphins": "mia",
        "New England Patriots": "nwe",
        "Baltimore Ravens": "bal", # idk something in 1999
        "New York Jets": "nyj",
        "Kansas City Chiefs": "kan",
        "Arizona Cardinals": "ari",
        "Minnesota Vikings": "min",
        "Green Bay Packers": "gnb",
        "New York Giants": "nyg",
        "Tennessee Titans": "ten",
        "Los Angeles Chargers": "lac", # San Diego Chargers until 2017
        "San Diego Chargers": "lac",
        "Las Vegas Raiders": "lvr",
        "Oakland Raiders": "lvr",
        "Tampa Bay Buccaneers": "tam",
        "Dallas Cowboys": "dal",
        "Seattle Seahawks": "sea",
        "Denver Broncos": "den"
    }

    # [This variable is unused] - Columns from the pre sliding window CSV file. Some columns listed here are removed in EDA
    original_cols = [
        # First Downs
        'H_First_Downs', 'V_First_Downs',
        
        # Basic Stats
        'H_Rush', 'V_Rush',
        'H_Yds', 'V_Yds',
        'H_TDs', 'V_TDs',
        'H_Cmp', 'V_Cmp',
        'H_Att', 'V_Att',
        'H_Yd', 'V_Yd',
        'H_TD', 'V_TD',
        'H_INT', 'V_INT',
        'H_Sacked', 'V_Sacked',
        'H_Sacked_Yards', 'V_Sacked_Yards',
        'H_Net_Pass_Yards', 'V_Net_Pass_Yards',
        'H_Total_Yards', 'V_Total_Yards',
        'H_Fumbles', 'V_Fumbles',
        'H_Lost', 'V_Lost',
        'H_Turnovers', 'V_Turnovers',
        'H_Penalties', 'V_Penalties',
        'H_Third_Down_Conv', 'V_Third_Down_Conv',
        'H_Fourth_Down_Conv', 'V_Fourth_Down_Conv',
        'H_Time_of_Possession', 'V_Time_of_Possession',
        
        # Passing Detailed
        'H_passing_att', 'V_passing_att',
        'H_passing_cmp', 'V_passing_cmp',
        'H_passing_int', 'V_passing_int',
        'H_passing_lng', 'V_passing_lng',
        'H_passing_sk', 'V_passing_sk',
        'H_passing_td', 'V_passing_td',
        # 'H_passing_yds', 'V_passing_yds',  # Removed in EDA
        
        # Receiving
        'H_receiving_lng', 'V_receiving_lng',
        # 'H_receiving_td', 'V_receiving_td', # Removed in EDA
        # 'H_receiving_yds', 'V_receiving_yds', # Removed in EDA
        
        # Rushing Detailed
        'H_rushing_att', 'V_rushing_att',
        'H_rushing_lng', 'V_rushing_lng',
        'H_rushing_td', 'V_rushing_td',
        'H_rushing_yds', 'V_rushing_yds',
        
        # Combined passing, rushing TD
        'H_passing_rushing_td', 'V_passing_rushing_td',
        
        # Defense Interceptions
        'H_def_interceptions_int', 'V_def_interceptions_int',
        # 'H_def_interceptions_lng', 'V_def_interceptions_lng', # Removed in EDA
        # 'H_def_interceptions_pd', 'V_def_interceptions_pd',
        'H_def_interceptions_td', 'V_def_interceptions_td',
        'H_def_interceptions_yds', 'V_def_interceptions_yds',
        
        # Defense Fumbles
        'H_fumbles_ff', 'V_fumbles_ff',
        'H_fumbles_fr', 'V_fumbles_fr',
        'H_fumbles_td', 'V_fumbles_td',
        'H_fumbles_yds', 'V_fumbles_yds',
        
        # Defense Tackles
        'H_sk', 'V_sk',
        'H_tackles_ast', 'V_tackles_ast',
        'H_tackles_comb', 'V_tackles_comb',
        # 'H_tackles_qbhits', 'V_tackles_qbhits',
        'H_tackles_solo', 'V_tackles_solo',
        # 'H_tackles_tfl', 'V_tackles_tfl',
        
        # ----------------- Kick & Punt returns are combined in EDA ----------------
        ## Kick Returns
        #'H_kick_returns_lng', 'V_kick_returns_lng',
        #'H_kick_returns_rt', 'V_kick_returns_rt',
        #'H_kick_returns_td', 'V_kick_returns_td',
        #'H_kick_returns_yds', 'V_kick_returns_yds',
        ## Punt Returns
        #'H_punt_returns_lng', 'V_punt_returns_lng',
        #'H_punt_returns_ret', 'V_punt_returns_ret',
        #'H_punt_returns_td', 'V_punt_returns_td',
        #'H_punt_returns_yds', 'V_punt_returns_yds',
        
        # Kick & Punt returns combined (Created as a result of EDA)
        'H_kick_punt_returns_lng', 'V_kick_punt_returns_lng',
        'H_kick_punt_returns_rt', 'V_kick_punt_returns_rt',
        'H_kick_punt_returns_td', 'V_kick_punt_returns_td',
        'H_kick_punt_returns_yds', 'V_kick_punt_returns_yds',
        
        # Punting/Scoring
        # 'H_punting_lng', 'V_punting_lng', # Removed in EDA
        
        'H_punting_pnt', 'V_punting_pnt',
        # 'H_punting_yds', 'V_punting_yds', # Removed in EDA
        'H_punting_avg', 'V_punting_avg',   # Created in EDA
        
        'H_scoring_fga', 'V_scoring_fga',
        # 'H_scoring_fgm', 'V_scoring_fgm', # Removed in EDA
        'H_scoring_fgp', 'V_scoring_fgp',   # Created in EDA
        
        'H_scoring_xpa', 'V_scoring_xpa',
        # 'H_scoring_xpm', 'V_scoring_xpm', # Removed in EDA 
        'H_scoring_xpp', 'V_scoring_xpp',   # Created in EDA
        
        # Final points, allowed points
        'H_Final', 'V_Final',
        'H_Final_Allowed', 'V_Final_Allowed',
        
        # Odds
        'H_start_odds', 'V_start_odds',
        'H_halftime_odds', 'V_halftime_odds'
    ]

    # Columns in the CSV post SlidingWindowNFL-1
    cont_cols = [
        'D_datediff',              # Days since last game (Home - visitor)
        
        # first downs
        'D_First_Downs',
        
        # Basic Stats
        'D_Rush',                  # Number of running plays attempted
        'D_Yds',                   # Yards gained through running plays
        'D_TDs',                   # Touchdowns scored via running plays
        'D_Cmp',                   # Completions (# of successful passes)
        'D_Att',                   # Attempts (# of passes thrown, completed or not)
        'D_Yd',                    # Yards (Yards the passes have covered)
        'D_TD',                    # Touchdowns
        'D_INT',                   # Interceptions
        'D_Sacked',                # Number of times quarterback was tackled behind line of scrimmage
        'D_Sacked_Yards',                 # Yards lost from sacks
        'D_Net_Pass_Yards',        # Net passing yards (total yds - yards lost due to sacks)
        'D_Total_Yards',           # Total yards gained (net pass yards + rushing yds)
        'D_Fumbles',               # Number of times ball was fumbled
        'D_Lost',                  # Number of times the team lost possession of the ball due to a fumble
        'D_Turnovers',             # Total number of turnovers, includes interceptions & fumbles lost
        'D_Penalties',             # Number of penalties committed by the team
        'D_Third_Down_Conv',       # 3rd down conversion percentage
        'D_Fourth_Down_Conv',      # 3rd down conversion percentage
        'D_Time_of_Possession',    # Time of possession in minutes
        
        
        # Passing Detailed
        'D_passing_att',           # Passes attempted
        'D_passing_cmp',           # Passes completed
        'D_passing_int',           # Interceptions thrown
        'D_passing_lng',           # Longest completed pass
        'D_passing_sk',            # Passing times sacked
        'D_passing_td',            # Passing touchdowns
        # 'D_passing_yds',           # Yards gained by passing
        
        # Receiving
        'D_receiving_lng',         # Longest reception
        # 'D_receiving_td',          # Receiving touchdowns
        # 'D_receiving_yds',         # Receiving yards
        
        # Rushing Detailed
        'D_rushing_att',           # Rushing attempts (sacks not included)
        'D_rushing_lng',           # Longest rushing attempt (sacks not included)
        'D_rushing_td',            # Rushing touchdowns
        'D_rushing_yds',           # Rushing yards
        
        # Defense interceptions
        'D_def_interceptions_int', # Passes intercepted on defense
        # 'D_def_interceptions_lng', # Longest interception returned
        'D_def_interceptions_td',  # Interceptions returned for touchdown
        'D_def_interceptions_yds', # Yards interceptions were returned
        
        # Defense fumbles
        'D_fumbles_ff',            # Num of times forced a fumble by the opposition recovered by either team
        'D_fumbles_fr',            # Fumbles recovered by player or team
        'D_fumbles_td',            # Fumbles recovered resulting in touchdown for receiver
        'D_fumbles_yds',           # Yards recovered fumbles were returned
        
        # Defense tackles
        'D_sk',                    # Sacks
        'D_tackles_ast',           # Assists on tackles
        'D_tackles_comb',          # Solo + ast tackles
        'D_tackles_solo',          # Tackles
    
        # ----------------- Kick & Punt returns are combined in EDA ----------------
        ## Kick Returns
        #'D_kick_returns_lng',      # Longest kickoff return
        #'D_kick_returns_rt',       # Kickoff returns 
        #'D_kick_returns_td',       # Kickoffs returned for a touchdown
        #'D_kick_returns_yds',      # Yardage for kickoffs returned
        ## Punt Returns
        #'D_punt_returns_lng',      # Longest punt return
        #'D_punt_returns_ret',      # Punts returned
        #'D_punt_returns_td',       # Punts returned for touchdown
        #'D_punt_returns_yds',      # Punts return yardage
        
        # Kick & Punt returns combined (Created as a result of EDA)
        #'kick_punt_returns_lng',   # Does not appear on final CSV (UMAP)
        #'kick_punt_returns_rt',    # Does not appear on final CSV (UMAP)
        #'kick_punt_returns_td',    # Does not appear on final CSV (UMAP)
        #'kick_punt_returns_yds',   # Does not appear on final CSV (UMAP)
        'kick_punt_umap_dim_1',  # Appears on final CSV (UMAP)
        'kick_punt_umap_dim_2',  # Appears on final CSV (UMAP)
        
        # Punting / Scoring
        # 'D_punting_lng',         # Longest punt
        
        'D_punting_pnt',           # Times punted
        # 'D_punting_yds',         # Total punt yardage
        'D_punting_avg',           # Total punt yardage / number of punts
        
        'D_scoring_fga',           # Field goals attempted
        # 'D_scoring_fgm',         # Field goals made
        'D_scoring_fgp',           # Field goals made / Field goals attempted
    
        'D_scoring_xpa',           # Extra points attempted
        # 'D_scoring_xpm',         # Extra points made
        'D_scoring_xpp',           # Extra pints made / Extra points attempted
        
        # Additional, calculated metrics
        'D_pythagorean',           # NFL variation of Bill James pythagorean expectation (from wikipedia)
        'D_start_odds',            # Gamebook odds
    ]

    drop_cols = [

        # Removed based off of correlation matrix results
        'D_passing_yds', 'D_receiving_yds', 'D_def_interceptions_lng', 'D_receiving_td',

        # Removed after composing meta-features from pairs of similar columns
        'D_scoring_fgm', 'D_scoring_xpm', 'D_punting_yds', 

        # Removed after applying UMAP
        'D_kick_returns_lng', 'D_kick_returns_rt', 'D_kick_returns_td', 'D_kick_returns_yds',
        'D_punt_returns_lng', 'D_punt_returns_ret', 'D_punt_returns_td', 'D_punt_returns_yds',

        # Created temporarily for UMAP
        'D_kick_punt_returns_lng', 'D_kick_punt_returns_rt', 'D_kick_punt_returns_td', 'D_kick_punt_returns_yds',
        
        # Removed from feature importance using Leshy & BoostAGroota
        'D_datediff', 'D_Third_Down_Conv', 'D_fumbles_ff', 'D_Net_Pass_Yards', 'D_TD'
    ]

    # Used in SlidingWindowNFL-1. Some variables here (like date) are required but not used as a model input
    track_cols = [
        # General
        'Date',  # Date
        
        # First Downs
        'First_Downs',
    
        # Basic Stats
        'Rush',
        'Yds',
        'TDs',
        'Cmp',
        'Att',
        'Yd',
        'TD',
        'INT',
        'Sacked',
        'Sacked_Yards',
        'Net_Pass_Yards',
        'Total_Yards',
        'Fumbles',
        'Lost',
        'Turnovers',
        'Penalties',
        'Third_Down_Conv',
        'Fourth_Down_Conv',
        'Time_of_Possession',
    
        # Passing Detailed
        'passing_att',
        'passing_cmp',
        'passing_int',
        'passing_lng',
        'passing_sk',
        'passing_td',
        # 'passing_yds', # Removed in EDA
    
        # Receiving
        'receiving_lng',
        # 'receiving_td', # Removed in EDA
        # 'receiving_yds', # Removed in EDA
    
        # Rushing Detailed
        'rushing_att',
        'rushing_lng',
        'rushing_td',
        'rushing_yds',
        
        # Combined passing, rushing TD 
        'passing_rushing_td', # TODO: REMOVE THIS 2/27
    
        # Defense Interceptions
        'def_interceptions_int',
        # 'def_interceptions_lng', # Removed in EDA
        # 'def_interceptions_pd',
        'def_interceptions_td',
        'def_interceptions_yds',
    
        # Defense Fumbles
        'fumbles_ff',
        'fumbles_fr',
        'fumbles_td',
        'fumbles_yds',
    
        # Defense Tackles
        'sk',
        'tackles_ast',
        'tackles_comb',
        # 'tackles_qbhits',
        'tackles_solo',
        # 'tackles_tfl',
    
        # ----------------- Kick & Punt returns are combined in EDA ----------------
        ## Kick Returns
        #'kick_returns_lng',
        #'kick_returns_rt',
        #'kick_returns_td',
        #'kick_returns_yds',
        ## Punt Returns
        #'punt_returns_lng',
        #'punt_returns_ret',
        #'punt_returns_td',
        #'punt_returns_yds',
        
        # Kick & Punt returns combined (Created as a result of EDA)
        'kick_punt_returns_lng',   # Does not appear on final CSV (UMAP)
        'kick_punt_returns_rt',    # Does not appear on final CSV (UMAP)
        'kick_punt_returns_td',    # Does not appear on final CSV (UMAP)
        'kick_punt_returns_yds',   # Does not appear on final CSV (UMAP)
        # 'kick_punt_umap_dim_1',  # Appears on final CSV (UMAP)
        # 'kick_punt_umap_dim_2',  # Appears on final CSV (UMAP)
    
        
        # Punting/Scoring
        # 'punting_lng', # Removed in EDA
        
        'punting_pnt',
        # 'punting_yds', # Removed in EDA
        'punting_avg',   # Created in EDA
        
        'scoring_fga',
        # 'scoring_fgm', # Removed in EDA
        'scoring_fgp',   # Created in EDA
        
        'scoring_xpa',
        # 'scoring_xpm', # Removed in EDA
        'scoring_xpp',   # Created in EDA
    
        # Final score, allowed
        'Final',
        'Final_Allowed',
        
        # Odds
        'start_odds',
        'halftime_odds'
    ]
    
    def __init__(self):
        pass
    
    def kelly_criterion(self, model_win_prob=0.6, prediction_decimal=0.55, fractional_odds=1.0):
        """
        returns a fraction (percent decimal) of current bankroll to wager
        
        See https://en.wikipedia.org/wiki/Kelly_criterion
        
        Args:
            model_win_prob (float): Model's win probability over the performance set
            prediction_decimal (float): Regression model's output for a certain bet. Between 0 and 1 with confidence threshold already applied to it
            fractional_odds (float): The sportsbook odds as fraction
        """
        model_prediction_decimal = abs(1 - 2*prediction_decimal) # if < 0.5
        if prediction_decimal > 0.5:
            model_prediction_decimal = abs(2*prediction_decimal - 1)
        
        # Average of model prediction odds and model win prob
        # print(f"prediction_decimal {prediction_decimal} model_prediction_decimal {model_prediction_decimal}")
        win_probability = sum([model_prediction_decimal, model_win_prob]) / 2
        bankroll_fraction = win_probability - ((1 - win_probability) / fractional_odds)
        return bankroll_fraction
        
    def map_losses(self, target_loss):
        """
        Created for optuna, returns one or a dict (target_loss=None) of the loss function
        to use based on the hyperparameter string (target_loss)
        """
        losses = {
            "MSELoss": nn.MSELoss(),
            "L1Loss": nn.L1Loss(),
            "SmoothL1Loss": nn.SmoothL1Loss()
        }
        if target_loss == None:
            return losses
        else:
            return losses[target_loss]

    
    def backtest_model(self, model, perf_conts, perf_y_col, perf_date_col, 
                       initial_capital=100, position_size=0.05, 
                       confidence_threshold=0.05, show_plot=True, max_won_odds=100):
        """
        Generated by claude, slightly modified. Shows a plot and returns a dict explaining model performance
        over len(perf_conts) samples
        """
        account_value = initial_capital
        x_axis = []
        y_axis = []
        wins = 0
        total_bets = 0
        max_value = initial_capital
        max_drawdown = 0

        # Get probabilities instead of rounded predictions
        probas = model.predict(perf_conts)

        mask = (probas < 0.5 - confidence_threshold) | (probas > 0.5 + confidence_threshold)
        predictions = np.where(mask, probas, np.nan)

        # Use numpy mask for nan values
        valid_mask = ~np.isnan(predictions)
        valid_predictions = predictions[valid_mask]
        valid_mask = valid_mask.flatten()
        perf_y_col = perf_y_col[valid_mask]
        perf_date_col = perf_date_col[valid_mask]

        # Calculate win probability for kelly_criterion
        true_values = perf_y_col[:,0].astype(np.int32)
        pred_values_int = np.rint(valid_predictions).flatten().astype(np.int32)
        pred_values = valid_predictions.flatten()
        model_win_prob = (1.0*(true_values == pred_values_int).sum()) / (true_values.shape[0])
        # print(f"model wn prob {model_win_prob}")
        
        current_date = ""
        current_day_bets = []
        for i in range(len(pred_values_int)):
            
            # Skip if odds are below max_won_odds
            actual_outcome = perf_y_col[i][0]
            won_odds = perf_y_col[i][1] if actual_outcome == 1 else perf_y_col[i][2]
            if won_odds > max_won_odds:
                # print(f"skipping {won_odds}")
                continue
                
            # Add to list if same day
            if current_date == perf_date_col[i][0]:
                current_day_bets.append(i)
                continue
            # Place bets and reset list
            else:
                current_date = perf_date_col[i][0]
            
            # Don't bet if no bets on current date
            if len(current_day_bets) == 0:
                continue
            # Prevent amount of bets that can be placed from exceeding the account value
            account_usable_cash = account_value
            
            # Adjust position size based on # of bets (10% if <= 10 bets, otherwise adjust)
            adjusted_position_size = min(0.1, (100.0 / len(current_day_bets)) / 100)
            # print(current_date)
            print(perf_date_col[i])
            while len(current_day_bets) != 0:
                bet_idx = current_day_bets.pop(0)

                # Determine prediction and actual outcome
                prediction = pred_values_int[bet_idx]
                actual = perf_y_col[bet_idx][0]
                won_odds = perf_y_col[bet_idx][1] if actual == 1 else perf_y_col[bet_idx][2]
                #if won_odds > max_won_odds:
                #    print(f"This shouldn't run.. won odds is {won_odds}")
                #    continue
                # print(f"won odss is {won_odds}")

                # Calculate position size (could be dynamic based on confidence)
                bet_size = account_value * adjusted_position_size
                kelly_res = 1 # self.kelly_criterion(model_win_prob, pred_values[bet_idx], won_odds)
                # Odds not favorable
                if kelly_res <=0:
                    continue
                # fall back to rest of $ if < suggested kelly amt
                bet_size = min(kelly_res * bet_size, account_usable_cash)
                
                # If the position exceeds the usable cash in a given day, continue
                account_usable_cash = account_usable_cash - bet_size
                if account_usable_cash < 0:
                    print(f"{current_date} exceed bet amt, skipping a bet")
                    continue
                    
                total_bets += 1
                #  kelly {kelly_res:.2f}
                print(f"{current_date}: w_odds:{won_odds:.2f} acct_val: {account_value:.2f} usable cash: {account_usable_cash:.2f} won: {actual == prediction}")
            
                # Calculate profit/loss
                if actual == prediction:
                    wins += 1
                    amt_won = bet_size * (won_odds - 1)
                else:
                    amt_won = -bet_size
                # print(f"{perf_y_col[i][0]} vs {valid_predictions[i]} odds:{won_odds}  ${amt_won}")
                # Update account value and track metrics
                account_value += amt_won
                max_value = max(max_value, account_value)
                current_drawdown = (max_value - account_value) / max_value
                max_drawdown = max(max_drawdown, current_drawdown)

                # Show chart
                x_axis.append(bet_idx)
                y_axis.append(account_value)
            current_day_bets = [i]

        # Calculate performance metrics
        win_rate = wins / total_bets if total_bets > 0 else 0
        roi = (account_value - initial_capital) / initial_capital

        # Plot results
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(x_axis, y_axis)
            plt.title(f'Backtest Results\nWin Rate: {win_rate:.2%} | ROI: {roi:.2%} | Max DD: {max_drawdown:.2%}')
            plt.xlabel('Number of Games')
            plt.ylabel('Account Value')
            plt.grid(True)
            plt.show()

        return {
            'final_value': account_value,
            'roi': roi,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_bets': total_bets
        }


    def backtest_model_custom_loss(self, performance_tensor_x, perf_date_col, 
                                   initial_capital=100, show_plot=True):
        """
        Generated by claude, slightly modified. Shows a plot and returns a dict explaining model performance
        over len(perf_conts) samples
        """
        account_value = initial_capital
        x_axis = []
        y_axis = []
        wins = 0
        no_bet = 0
        total_bets = 0
        max_value = initial_capital
        max_drawdown = 0
        
        current_date = ""
        current_day_bets = []
        for i in range(len(performance_tensor_x)):
            # Add to list if same day and < 10 bets
            if current_date == perf_date_col[i][0] and len(current_day_bets) < 10:
                current_day_bets.append(i)
                continue
            # Ignore if > 10th bet on the current day
            elif current_date == perf_date_col[i][0]:
                continue
            # Place bets and reset list
            else:
                current_date = perf_date_col[i][0]
            
            # Don't bet if no bets on current date
            if len(current_day_bets) == 0:
                continue
            # Prevent amount of bets that can be placed from exceeding the account value
            account_usable_cash = account_value
            
            # print(perf_date_col[i])
            while len(current_day_bets) != 0:
                bet_idx = current_day_bets.pop(0)
                # Calculate position size (could be dynamic based on confidence)
                amt_after_bet = account_value * performance_tensor_x[bet_idx]

                total_bets += 1
                #  kelly {kelly_res:.2f}
                print(f"{current_date}: acct_val: {account_value:.2f} usable cash: {account_usable_cash:.2f} won: {amt_after_bet > account_value}")

                if amt_after_bet > account_value:
                    wins = wins + 1
                if amt_after_bet == account_value:
                    no_bet = no_bet + 1
                # Update account value and track metrics
                account_value = amt_after_bet
                max_value = max(max_value, account_value)
                current_drawdown = (max_value - account_value) / max_value
                max_drawdown = max(max_drawdown, current_drawdown)

                # Show chart
                x_axis.append(bet_idx)
                y_axis.append(account_value)
            current_day_bets = [i]

        # Calculate performance metrics
        win_rate = wins / (total_bets-no_bet) if total_bets > 0 else 0
        roi = (account_value - initial_capital) / initial_capital

        # Plot results
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(x_axis, y_axis)
            plt.title(f'Backtest Results\nWin Rate: {win_rate:.2%} | ROI: {roi:.2%} | Max DD: {max_drawdown:.2%}')
            plt.xlabel('Number of Games')
            plt.ylabel('Account Value')
            plt.grid(True)
            plt.show()

        return {
            'final_value': account_value,
            'roi': roi,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_bets': total_bets
        }
    
    
    def old_backtest_model(self, predictions, perf_y_col):
        """
        My original backtest_model code. Used as a sanity check for backtest_model
        
        Assumes the following
        Column 0: H_Won
               1: H_start_odds
               2: V_start_odds
        """
        account_value = 100
        position_size = 0.1
        x_axis = []
        y_axis = []
        for i in range(0,len(predictions)):
            amt_won = -(position_size * account_value)
            won_odds = perf_y_col[i][1] if perf_y_col[i][0] == 1 else perf_y_col[i][2]
            # correct prediction
            if perf_y_col[i][0] == predictions[i]:
                amt_won = abs(amt_won) * (won_odds - 1)
            # Update account value
            # print(f"{perf_y_col[i][0]} vs {predictions[i]} odds:{won_odds}  ${amt_won}")
            account_value = account_value + amt_won
            x_axis.append(i)
            y_axis.append(account_value)
        plt.plot(x_axis, y_axis)
        plt.show()
        
        
        
# ----------------------------------------------------
# ----------- "SlidingWindowNFL" Utils ---------------
# ----------------------------------------------------
    def get_track_dict(self, df):
        """
        df from reading combined.csv
        """
        track_dict = {}

        for row in df.itertuples():
            # year = row.Date.split('-')[0]
            year = row.Season
            home_team = row.Home_Team
            visitor_team = row.Visitor_Team

            # Home or visitor team has < minimum_window total games
            for col in self.track_cols:
                home_column_name = f'{year}_{home_team}_{col}'
                visitor_column_name = f'{year}_{visitor_team}_{col}'

                # Home team
                home_col = col if col == 'Date' else 'H_' + col
                if home_column_name in track_dict:
                    track_dict[home_column_name].append(getattr(row, home_col))
                else:
                    track_dict[home_column_name] = [getattr(row, home_col)]

                # Visitor team
                visitor_col = col if col == 'Date' else 'V_' + col
                if visitor_column_name in track_dict:
                    track_dict[visitor_column_name].append(getattr(row, visitor_col))
                else:
                    track_dict[visitor_column_name] = [getattr(row, visitor_col)]
        return track_dict
        
        
    def get_current_date_games(self, currentSeason, dateOffset):
        """
        Retrieves the list of games for the current date provided the currentSeason
        
        If error occurrs, will print a message and return an empty array
        """
        currentDateGames = []
        # Example: https://www.pro-football-reference.com/years/2022/games.htm
        gamesList = r.get('https://www.pro-football-reference.com/years/' + str(currentSeason) + '/games.htm')
        if gamesList.status_code != 200:
            print("Got invalid status code 1")
            return []
        gList = bs(gamesList.text, features='html.parser')

        boxscoreLinks = gList.select('td[data-stat="boxscore_word"] a')
        hrefs = [link['href'] for link in boxscoreLinks if 'href' in link.attrs]

        # iterate through each url and ignore until you reach the current date
        for i in range(0, len(hrefs)):
            href = hrefs[i]
            formattedDate = f"{href[11:15]}-{href[15:17]}-{href[17:19]}"
            formattedDateObj = datetime.strptime(formattedDate, '%Y-%m-%d').date()
            currentDate = datetime.now().date() + timedelta(days=dateOffset)
            if formattedDateObj != currentDate:
                continue

            boxscoreLinks = gList.select('td[data-stat="boxscore_word"] a')
            visitorList = gList.select('td[data-stat="winner"] a')
            homeList = gList.select('td[data-stat="loser"] a')
            bsLen = len(boxscoreLinks)

            if bsLen != len(visitorList) or bsLen != len(homeList):
                print("One of boxscore, visitor, or home list doesn't match the others")
                return []
            currentDateGames.append({'Season': currentSeason, 'Home_Team': self.team_abbrv[homeList[i].text].upper(), 'Visitor_Team': self.team_abbrv[visitorList[i].text].upper(), 'Date':formattedDate})
        return currentDateGames

    def getMLBOddsSharkData(isoDate):
        """
        Get odds data from Odds Shark API
        """
        url = f"https://www.oddsshark.com/api/scores/mlb/{isoDate}?_format=json"
        print(url)
        response = r.get(url)
    
        # Return early if error
        if response.status_code != 200:
            print(f"getOddsSharkData: Error fetching data: {response.status_code}")
            return
        
        scores = response.json()["scores"]
        results = []
        for game in scores:
            results.append({
                    "Date": isoDate,
                    "Rot": game["teams"]["home"]["rotation"],
                    "Home_Team": game["teams"]["home"]["names"]["abbreviation"],
                    "Home_Odds": game["teams"]["home"]["moneyLine"],
                    "Visitor_Team": game["teams"]["away"]["names"]["abbreviation"],
                    "Visitor_Odds": game["teams"]["away"]["moneyLine"],
                })
        return results
    
            
        
        
        
        
        

    
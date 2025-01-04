import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime, timedelta
import math
import torch.nn as nn
import requests as r
from bs4 import BeautifulSoup as bs


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
    track_cols = [
        # General
        'Date',  # Date

        # First Downs
        'First_Downs',

        # Rushing
        'Rush',
        'Yds',
        'TDs',

        # Passing
        'Cmp',
        'Att',
        'Yd',
        'TD',
        'INT',
        'Sacked',
        'Yards',
        'Net_Pass_Yards',

        # Total Yards
        'Total_Yards',

        # Fumbles
        'Fumbles',
        'Lost',
        'Turnovers',

        # Penalties
        'Penalties',
        'Yards',

        # Third Down Conversions
        # 'Third_Down_Conv',

        # Fourth Down Conversions
        # 'Fourth_Down_Conv',

        # Time of Possession
        # 'Time_of_Possession',

        # Passing Detailed
        'passing_att',
        'passing_cmp',
        'passing_int',
        'passing_lng',
        'passing_sk',
        'passing_td',
        'passing_yds',

        # Receiving
        'receiving_lng',
        'receiving_td',
        'receiving_yds',

        # Rushing Detailed
        'rushing_att',
        'rushing_lng',
        'rushing_td',
        'rushing_yds',

        # Defense Interceptions
        'def_interceptions_int',
        'def_interceptions_lng',
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

        # Kick Returns
        'kick_returns_lng',
        'kick_returns_rt',
        'kick_returns_td',
        'kick_returns_yds',

        # Punt Returns
        'punt_returns_lng',
        'punt_returns_ret',
        'punt_returns_td',
        'punt_returns_yds',

        # Punting
        'punting_lng',
        'punting_pnt',
        'punting_yds',
        'scoring_fga',
        'scoring_fgm',
        'scoring_xpa',
        'scoring_xpm',

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
            if current_date == perf_date_col[i]:
                current_day_bets.append(i)
                continue
            # Place bets and reset list
            else:
                current_date = perf_date_col[i]
            
            # Don't bet if no bets on current date
            if len(current_day_bets) == 0:
                continue
            # Prevent amount of bets that can be placed from exceeding the account value
            account_usable_cash = account_value
            
            # Adjust position size based on # of bets (10% if <= 10 bets, otherwise adjust)
            adjusted_position_size = min(0.1, 100.0 / len(current_day_bets))
            print(current_day_bets)
            # print(current_date)
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
                # print(f"{current_date[0]} kelly {kelly_res:.2f} w_odds:{won_odds:.2f} acct_val: {account_value:.2f} usable cash: {account_usable_cash:.2f} won: {actual == prediction}")
            
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
        
        
        
        
        
        
        

    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import math
import torch.nn as nn
import requests as r
from bs4 import BeautifulSoup as bs
import umap


class MLBUtils():
    teams = {
        'ARI': 'Arizona Diamondbacks',
        'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles',
        'BOS': 'Boston Red Sox',
        'CHA': 'Chicago White Sox',
        'CHN': 'Chicago Cubs',
        'CIN': 'Cincinnati Reds',
        'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies',
        'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros',
        'KCA': 'Kansas City Royals',
        'ANA': 'Los Angeles Angels',
        'LAN': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins',
        'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins',
        'NYA': 'New York Yankees',
        'NYN': 'New York Mets',
        'OAK': 'Oakland Athletics',
        'ATH': 'Athletics',
        'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates',
        'SDN': 'San Diego Padres',
        'SFN': 'San Francisco Giants',
        'SEA': 'Seattle Mariners',
        'SLN': 'St. Louis Cardinals',
        'TBA': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers',
        'TOR': 'Toronto Blue Jays',
        'WAS': 'Washington Nationals'
    }

    """
    Want: 
        An array > 1 elements for PCC formula
        A smaller array for kelly_criterion
        
    
    
    x: 1d array of predictions between -1 and 1 where negative number means visitor predicted to win
    y: ['H_Won', 'H_start_odds', 'V_start_odds']
    pearson_multiplier: constant to multiply the pearson correlation coefficient's result by
    max_bet_pct: max percentage allowed per bet.
    return_res_array: Skip the torch.cumprod part
    """
    def custom_criterion(self, x, y, pearson_multiplier=0.5, max_bet_pct=0.1, return_res_array=False):
        # ------------------------------------------------
        # Preliminary calculations
        # ------------------------------------------------
        # acct_value = 100 # Preset account value
        batch_size = len(x)
        h_start_odds = y[:,1]
        v_start_odds = y[:,2]
        h_won = y[:,0]
        y_decimal_odds = torch.where(x > 0, h_start_odds, v_start_odds) # Decimal odds for model's predicted outcome
        y_prob = 1 / y_decimal_odds                  # Implied Probability (regardless of correct prediction)
        x_H_Won = torch.round(torch.sigmoid(20 * x)) # H_won for predicted bets (Converts model's -1 to 1 range to 0 to 1)
                                                     # Sigmoid so that it's differentiable. The 20 is arbitrarily large number
        y_incorrect_prediction = torch.abs((x_H_Won - h_won))        # 1 if wrong bet, otherwise 0. Used to reset kelly when wrong
        y_incorrect_prediction_mult_two = 2 * y_incorrect_prediction   # 2 if wrong bet, 0 if correct
    
        #x = torch.abs(x)         # OLD VERSION
        x_prob = torch.where(x > 0, (x + 1) / 2, (1 - x) / 2)
        x = x_prob                # x is now the implied probability(?) of your prediction.
                                  # It's a number between 0 and 1 (formerly -1 and 1) representing the model's predicted probability of a win.
                                  # This now only shows the probability. Not whether it was correct & not for which side (home vs visitor)
    
        
        # ------------------------------------------------
        # 1. Calculate the Pearson Correlation Coefficient
        #    Currently includes cases where predicted wrong
        #    ^ This is filtered out after step 2
        # ------------------------------------------------
        n = x.size(0)
        sum_x = torch.sum(x)
        sum_x_squared = torch.sum(x**2)
        sum_y = torch.sum(y_prob)
        sum_y_squared = torch.sum(y_prob**2)
        sum_pow_x = torch.sum(x**2)
        sum_pow_y = torch.sum(y_prob**2)
        x_mul_y = torch.mul(x, y_prob)
        sum_x_mul_y = torch.sum(x_mul_y)
    
        
        # PCC Formula (eps to avoid NaN)
        eps = 1e-8
        pcc_numerator = n * sum_x_mul_y - sum_x * sum_y
        pcc_denominator_one = torch.sqrt(n * sum_pow_x - sum_x_squared + eps)
        pcc_denominator_two = torch.sqrt(n * sum_pow_y - sum_y_squared + eps)
        pcc = pcc_numerator / (pcc_denominator_one * pcc_denominator_two + eps)
        pcc = pearson_multiplier * torch.abs(pcc)
    
        
        # ------------------------------------------------
        # 2. Calculate the kelly criterion
        #    Entirely wrong predictions are negated and kept in "incorrect_bets" (pcc not applied to wrong predictions)
        #    Correct predictions are kept in "correct_bets". Pcc is applied to this & stored in pcc_adjusted_correct_bets
        #    Possible issue: This always bets max_bet_pct
        #    The result is cumulatively calculated. i.e. The sum of the previous values are used to calculate the next one
        # ------------------------------------------------
        # kelly_criterion = x - ((1 - x) / y_decimal_odds)  # OLD VERSION
        kelly_criterion = x - ((1 - x) / (y_decimal_odds - 1))
        bet_multiplier = torch.clamp(kelly_criterion, min=0)   # Kelly results that are negative are ignored
        bet_multiplier = bet_multiplier*max_bet_pct            # Scale down the bets to the maximum allowed percentage per bet
    
    
        # 4/5/25 adjustment of kelly
        #    Want to use cumprod. Cumsum does nothing and is the same as torch.sum in this scenario?
        #    Basically start with max_bet_pct and return as if you made the bets sequentially
        correct_bet_multiplier = bet_multiplier - (bet_multiplier * y_incorrect_prediction)          # Correct bets after kelly. Bet multiplier or 0
        if not return_res_array:
            correct_bet_multiplier = correct_bet_multiplier * (1 - pcc)                              # "correct_bet_multiplier" penalized by pcc
        assert torch.all(correct_bet_multiplier <= max_bet_pct), "Correct bet mult. can't exceed max bet pct"
    
        correct_bet_multiplier = correct_bet_multiplier * (y_decimal_odds - 1)                       # Bet multiplier taking market odds into account
        incorrect_bet_multiplier = bet_multiplier - (bet_multiplier * y_incorrect_prediction_mult_two) # Negative numbers are incorrect bets
        incorrect_bet_multiplier = torch.clamp(incorrect_bet_multiplier, max=0)                      # Restrict to 0 or negative
        combined_bet_multiplier = correct_bet_multiplier + incorrect_bet_multiplier                  # Combine correct & incorrect bet multipliers
        combined_bet_multiplier = combined_bet_multiplier + 1                                        # Converts to format friendly to cumprod
                                                                                                     # Ex: loss=-0.3, profit=0.3 --> loss=0.7, profit=1.3
    
        assert torch.all((x >= 0) & (x <= 1)), "Probabilities must be between 0 and 1"
        assert torch.all(y_decimal_odds > 1), "Decimal odds must be greater than 1"
        assert torch.all(kelly_criterion <= 1), "Kelly Criterion cannot be greater than 1"
        assert torch.all(incorrect_bet_multiplier >= -max_bet_pct), "Incorrect bet mult. can't exceed max bet pct"
        
        # ------------------------------------------------
        # Combine & Return
        #     Negate everything for Adam & optuna
        # ------------------------------------------------
        if return_res_array:
            return combined_bet_multiplier
            
        # Prepend max_bet_pct to the tensor before torch.cumprod
        res = torch.sum(combined_bet_multiplier) / batch_size
        # print(res)
        return -res

    def americanToDecimal(self, odds):
        """
        Convert american to decimal odds.
        """
        if odds < 0:
            return 1 + 100 / abs(odds)
        return 1 + odds / 100

    def get_team_name(self, abbreviation):
        return self.teams.get(abbreviation, 'Unknown team')


    def backtest_model_custom_loss(self, performance_tensor_x, perf_date_col, 
                                   initial_capital=100, show_plot=True, home_teams=[], visitor_teams=[], probas=[]):
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
        
        current_date = perf_date_col[0][0]
        current_day_bets = []
        for i in range(len(performance_tensor_x)):
            # Add to list if same day and < 10 bets
            if current_date == perf_date_col[i][0] and len(current_day_bets) < 20:
                current_day_bets.append(i)
                continue
            # Ignore if > 20th bet on the current day
            elif current_date == perf_date_col[i][0]:
                continue
            # Place bets and reset list
            
            # Don't bet if no bets on current date
            if len(current_day_bets) == 0:
                continue
            # Prevent amount of bets that can be placed from exceeding the account value
            account_usable_cash = account_value
            
            # print(perf_date_col[i])
            num_current_day_bets = len(current_day_bets)
            
    
            while len(current_day_bets) != 0:
                bet_idx = current_day_bets.pop(0)
                # Calculate position size (could be dynamic based on confidence)
                amt_after_bet = account_value * performance_tensor_x[bet_idx]
    
                total_bets += 1
                #  kelly {kelly_res:.2f}
                # print(f"{current_date}: acct_val: {account_value:.2f} usable cash: {account_usable_cash:.2f} won: {amt_after_bet > account_value}")
    
                winning_team = visitor_teams[bet_idx]
                losing_team = home_teams[bet_idx]
                if probas[bet_idx] > 0:
                    winning_team = home_teams[bet_idx]
                    losing_team = visitor_teams[bet_idx]
    
                print(f"{current_date} - Pred: {winning_team} won against {losing_team}. Correct: {amt_after_bet > account_value} {amt_after_bet} ")
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
            current_date = perf_date_col[i][0]
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




import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn as nn

class NFLUtils():
    """
    kelly_criterion-> Used by backtest_model to calculate optimal position size
    map_losses     -> Used by optuna in NFL_ANN to map b/t loss functions
    backtest_model -> Calculates and charts a backtest against the performance set
    """
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

    
    def backtest_model(self, model, perf_conts, perf_y_col, initial_capital=100, position_size=0.05, 
                       confidence_threshold=0.05, show_plot=True):
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

        # Calculate win probability for kelly_criterion
        true_values = perf_y_col[:,0].astype(np.int32)
        pred_values_int = np.rint(valid_predictions).flatten().astype(np.int32)
        pred_values = valid_predictions.flatten()
        model_win_prob = (1.0*(true_values == pred_values_int).sum()) / (true_values.shape[0])
        # print(f"model wn prob {model_win_prob}")
        
        for i in range(len(pred_values_int)):
            total_bets += 1

            # Determine prediction and actual outcome
            prediction = pred_values_int[i]
            actual = perf_y_col[i][0]
            won_odds = perf_y_col[i][1] if actual == 1 else perf_y_col[i][2]
            # print(f"won odss is {won_odds}")
            
            # Calculate position size (could be dynamic based on confidence)
            bet_size = account_value * position_size
            kelly_res = self.kelly_criterion(model_win_prob, pred_values[i], won_odds)
            bet_size = kelly_res * bet_size
            # print(f"kelly_res: {kelly_res} model_win_prob: {model_win_prob}  won_odds:{won_odds} account_val: {account_value}")
            
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

            x_axis.append(i)
            y_axis.append(account_value)
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

    
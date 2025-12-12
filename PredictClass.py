from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import pickle
import requests
from elopy import Implementation
from scipy.optimize import curve_fit


class PredictClass():
    def predict(self, numRecords=1000):
        # %%
        
        # Import & drop duplicates for ttElite
        dfElite = pd.read_csv('/Users/forbesjon2/drive/Notes/ML/Pytorch/ttennisData/TTElite.csv')
        dfElite = dfElite.drop_duplicates(keep='last')

        # Import & drop duplicates for ttCup
        df = pd.read_csv("/Users/forbesjon2/drive/Notes/ML/Pytorch/ttennisData/TTCup.csv")
        df = df.drop_duplicates(keep='last')

        df = pd.concat([df, dfElite])
        print(f"predictImport: {df.shape}")

        df = df.fillna(0) # G4+ have 

        # ------- Create new columns for df -------
        # Combine P1 and P2 scores for each game
        for i in range(1, 6):
            df[f'G{i}_Total'] = df[f'P1_G{i}'] + df[f'P2_G{i}']

        # Player 1 total score from G1 to G5
        df['Total_P1'] = df[[f'P1_G{i}' for i in range(1, 6)]].sum(axis=1)
        df['Total_Avg_P1'] = df['Total_P1']

        # Player 2 total score from G1 to G5
        df['Total_P2'] = df[[f'P2_G{i}' for i in range(1, 6)]].sum(axis=1)
        df['Total_Avg_P2'] = df['Total_P2']

        # Sum of combined scores from G1 to G5
        df['Total_Score'] = df[[f'G{i}_Total' for i in range(1, 6)]].sum(axis=1)
        df['Over_74'] = df['Total_Score'] > 74
        df['G5'] = df['G5_Total'] > 0
        df['G4'] = (df['G4_Total'] > 0) & (df['G5_Total'] == 0)
        df['G45'] = (df['G4_Total'] > 0) | (df['G5_Total'] > 0)
        df['G34'] = (df['G5_Total'] == 0) & (df['Total_Score'] > 0)

        for i in range(1,6):
            df.rename(columns={f'P1_G{i}': f'G{i}_P1', f'P2_G{i}': f'G{i}_P2'}, inplace=True)
        
        df.rename(columns={'P1_Total': 'Total_P1', 'P2_Total': 'Total_P2'}, inplace=True)

        # df['No_Odds'] = (df['Odds_P1'] + df['Odds_P2'] == 0)
        # print(df['No_Odds'].sum())

        df['Total_Allowed_P1'] = df['Total_P2']
        df['Total_Allowed_P2'] = df['Total_P1']
        df['Sets_Allowed_P1'] = df['Sets_P2']
        df['Sets_Allowed_P2'] = df['Sets_P1']

        df['Win_P1'] = df['Total_P1'] > df['Total_P2']
        df['Win_P2'] = df['Total_P2'] > df['Total_P1']

        df = df.sort_values(by='Date')

        # %%
        cont_cols = [
            'Date',
            'Player1',
            'Player2',
            'Sets_P1',
            'Sets_P2',

            'G1_P1',
            'G1_P2',
            'G2_P1',
            'G2_P2',
            'G3_P1',
            'G3_P2',
            'G4_P1',
            'G4_P2',
            'G5_P1',
            'G5_P2',

            'Win_P1',
            'Win_P2',

            #'G34_P1',
            #'G34_P2',
            
        #    'No_Odds', # Removed from EDA
            'Total_P1', # Unadjusted total (careful re: leakage)
            'Total_P2', # Unadjusted total (careful re: leakage)
            'Total_Avg_P1', # Non-leakage of total
            'Total_Avg_P2', # Non-leakage of total
            'Total_Score', # Unadjusted total (careful re: leakage)
            'Total_Allowed_P1',
            'Total_Allowed_P2',
            'Sets_Allowed_P1',
            'Sets_Allowed_P2',
        ]

        track_cols = [
            'Date',
            'Sets',
            'Sets_Allowed',
            'G1',
            'G2',
            'G3',
            'G4',
            'G5',
            
            #'G45',
            #'G34',
            'Win',
            
            'Odds',
            'Total',
            'Total_Avg',
            'Total_Allowed',
        ]
        y_col = ['H_won']

        # Convert columns to int (it originally was int before odds data was added)
        for col in cont_cols:
            df[col] = df[col].astype(np.int64)
        cont_cols = cont_cols + ['Over_74','G5','G4', 'G34', 'G45', 'Odds_P1','Odds_P2','Player1_Name', 'Player2_Name']

        dfOriginal = df
        df = df[cont_cols]

        # %% [markdown]
        # ### 2. Create a dict to track the track_cols array
        # create another dict to track previous games for each team during the year
        # 

        # %%
        df.head()

        # %% [markdown]
        # ### Create dfUpcoming with upcoming records, df with past games

        # %%
        # Copy of df without upcoming data (use the sum of last 10 columns as a judge)
        
        # NEW FOR local predictClass implementation: remove upcoming data entirely
        last10 = df.iloc[:,-17:-14]
        rowSum = last10.sum(axis=1)
        df = df[rowSum != 0]

        # NEW FOR local predictClass implementation: Ignore upcoming dat, manually reset numRecords
        # print(f"df shape: {df.shape}")
        df.loc[df.index[-numRecords:], 'Total_P1'] = 0
        df.loc[df.index[-numRecords:], 'Total_P2'] = 0
        df.loc[df.index[-numRecords:], 'Total_Avg_P1'] = 0
        df.loc[df.index[-numRecords:], 'Total_Avg_P2'] = 0
        dfUpcoming = df
        last10 = dfUpcoming.iloc[:,-17:-14]
        rowSum = last10.sum(axis=1)
        
        df = dfUpcoming[rowSum != 0]
        # print(dfUpcoming.tail())

        print(f"dfUpcoming.shape: {dfUpcoming.shape}")

        # %%
        track_dict = {}
        elopy = Implementation()

        def addToTrackDict(column_name, value):
            """
            Adds value to track_dict. Handles cases where it doesn't exist
            """
            if column_name in track_dict:
                track_dict[column_name].append(value)
            else:
                track_dict[column_name] = [value]


        for row in df.itertuples():
            year = datetime.fromtimestamp(row.Date).year # row.Date.split('-')[0]
            # year = row.Season
            home_team = row.Player1
            visitor_team = row.Player2

            # ELO calculation
            if not elopy.contains(home_team):
                elopy.addPlayer(home_team)
            if not elopy.contains(visitor_team):
                elopy.addPlayer(visitor_team)
            winning_team = home_team if getattr(row, 'Win_P1') == 1 else visitor_team
            elopy.recordMatch(home_team, visitor_team, winner=winning_team)
            addToTrackDict(f'{year}_{home_team}_elo', elopy.getPlayerRating(home_team))
            addToTrackDict(f'{year}_{visitor_team}_elo', elopy.getPlayerRating(visitor_team))
            
            # Home or visitor team has < minimum_window total games
            for col in track_cols:
                if col in ['elo']:
                    continue
                home_column_name = f'{year}_{home_team}_{col}'
                visitor_column_name = f'{year}_{visitor_team}_{col}'
                
                # Home team
                home_col = col if col == 'Date' else col + '_P1'
                addToTrackDict(home_column_name, getattr(row, home_col))
                
                # Visitor team
                visitor_col = col if col == 'Date' else col + '_P2'
                addToTrackDict(visitor_column_name, getattr(row, visitor_col))

        # %%
        print(df.shape)
        print(dfUpcoming.shape)
        onlyUpcoming = dfUpcoming.merge(df, how='outer', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
        print(onlyUpcoming.shape)
        # %% [markdown]
        # ### Check and remove missing data

        # %%
        df = dfUpcoming
        df = onlyUpcoming

        # Create df according to best tested params (note: this df contains only the upcoming data)
        print("- TEST df shape - ")
        print(df.shape)
        dfOriginal = df
        g5_df = self.createDf(track_dict, track_cols, df, 10, 100, 15)
        print(g5_df.shape)
        df = dfOriginal
        hwon_df = self.createDf(track_dict, track_cols, df, 10, 100, 30)
        print(hwon_df.shape)

        # G5 + accuracy

        # -- G5 --
        # Name: g5_dec1
        # 1481/10k (14.8%), accuracy: 0.33
        # 10, 100, 15 for G5
        # no adjustment except perf_conts_shift=0.19
        g5_x_col = ['G4_P1', 'G4_P2', 'G5_P1', 'G5_P2', 'elo_P1', 'elo_P2', 'Total_Avg_P1', 'Total_Avg_P2', 'Total_Allowed_P1', 'Total_Allowed_P2', 'D_Odds', 'G1_P1', 'pythagorean_P1', 'pythagorean_P2']
        g5_mask = self.makePrediction(g5_df, g5_x_col, "g5_dec1", 1.0, None, True, 0.0, 0.19, "G5")

        # -- H_won --
        # Name: hwon_dec1
        # 3,560/10k (35.6%), accuracy 0.72
        # 10, 100, 30 for H_won
        # no adjustment except limit=0.13
        accuracy_x_col = ['G1_P2', 'D_G1', 'D_G2', 'D_G3', 'Win_P1', 'pythagorean_P1', 'pythagorean_P2', 'elo_P1', 'elo_P2', 'Total_Allowed_P1', 'Total_Allowed_P2', 'Sets_Allowed_P2', 'D_Odds', 'Sets_P2', 'G2_P2', 'Win_P2', 'Sets_Allowed_P1']
        self.makePrediction(hwon_df, accuracy_x_col, "hwon_dec1", 0.0, g5_mask, False, 0.13, 0.0, "Accuracy")



        # # 512/2k (25.6%) with limit 0, perf_conts_shift = 0.17 at 0.384 accuracy
        # g5_x_col = ['G5_P1', 'Total_Allowed_P2', 'G5_P2', 'elo_P2', 'Total_Allowed_P1', 'G4_P1', 'Total_Avg_P2', 'D_Odds', 'Total_Avg_P1', 'elo_P1', 'D_G2', 'G1_P1', 'G3_P1', 'pythagorean_P1', 'pythagorean_P2']
        # g5_mask = self.makePrediction(df, g5_x_col, "g5_nov8", 1.0, None, True, 0.0, 0.17, "G5")
        # accuracy_x_col = ['D_Odds', 'pythagorean_P1', 'pythagorean_P2', 'elo_P2', 'elo_P1', 'D_G3', 'D_G1', 'Sets_P2', 'Win_P1', 'Total_Allowed_P1', 'D_G2', 'Win_P2', 'Total_Allowed_P2', 'Sets_P1']
        # # accuracy_mask is on dfUpcoming
        # # 195/2k (9.75%) with G5, ROC AUC (higher is better):  0.6892887205387205 Accuracy 0.6820512820512821
        # self.makePrediction(df, accuracy_x_col, "accuracy_nov8", 0.0, g5_mask, False, 0.12, 0.0, "Accuracy")
        # # break even odds +290

        # Over / under 74.5



    def createDf(self, track_dict, track_cols, df, minimum_window, max_window, ema_span):
        """
        track_dict: Instance of track_dict. Unchanged by this function, can use across multiple calls to createDf
        track_cols: Instance of track_cols. Unchanged by this function, can use across multiple calls to createDf
        df: instance of dataframe (edits in place, keep an original 'before' copy outside of this
        minimum_window: minimum number of records to get the average of
        max_window: The maximum number of records to get the average of
        ema_span: the exponential moving average span.
        """
        indices_to_drop = []
        current_count = 0
        for row in df.itertuples():
            if current_count % 500 == 0:
                print(f'{current_count}/{df.shape[0]}')
            current_count = current_count + 1
            index = row.Index
            year = datetime.fromtimestamp(row.Date).year # row.Date.split('-')[0]
            # year = row.Season
            home_team = row.Player1
            visitor_team = row.Player2
            # Home team min window
            home_date_column = f'{year}_{home_team}_Date'
            visitor_date_column = f'{year}_{visitor_team}_Date'
        
            # Drop indice if not enough data
            # if len(track_dict[home_date_column]) <= minimum_window or len(track_dict[visitor_date_column]) <= minimum_window:
            # Drop index if column is missing OR not enough data (12/4/25)
            if (home_date_column not in track_dict.keys() or 
                visitor_date_column not in track_dict.keys() or
                len(track_dict[home_date_column]) <= minimum_window or 
                len(track_dict[visitor_date_column]) <= minimum_window):
                indices_to_drop.append(index)
                continue
        
            # Current row is older than Home team at min_window
            if row.Date <= track_dict[home_date_column][minimum_window]:
                indices_to_drop.append(index)
                continue
            # Current row is older than Visitor team at min_window
            if row.Date <= track_dict[visitor_date_column][minimum_window]:
                indices_to_drop.append(index)
                continue
        
            # 9/15/25 - Future games have no index, get last index if not found.
            # home_date_index = track_dict[home_date_column].index(row.Date)
            # home_date_index = track_dict[home_date_column].index(row.Date) if row.Date in track_dict[home_date_column] else len(track_dict[home_date_column]) - 1
            home_date_index = len(track_dict[home_date_column]) - 1

            # visitor_date_index = track_dict[visitor_date_column].index(row.Date)
            # visitor_date_index = track_dict[visitor_date_column].index(row.Date) if row.Date in track_dict[visitor_date_column] else len(track_dict[visitor_date_column]) - 1
            visitor_date_index = len(track_dict[visitor_date_column]) - 1
            # print(f'H: {home_date_index} V: {visitor_date_index}')
        
            # print("----- Track Cols -----")
            # Update df to have average for each track_cols (Ignoring 'Date', the 1-2nd item)
            for col in track_cols:
                if col in ['Date', 'Odds', 'Total']:
                    continue
                else:
                    # print(col)
                    # Convert to float
                    if df.dtypes[col + '_P1'] == 'int64' or df.dtypes[col + '_P2'] == 'int64':
                        df = df.astype({f'{col}_P1': 'float64', f'{col}_P2': 'float64'})
                    # Update df to have average for each track_cols (Ignoring 'Date', .. items)
                    # Update home
        
                    start, end = self.calcStartEndIdx(home_date_index, max_window)
                    home_col_list = track_dict[f'{year}_{home_team}_{col}'][start:end]
                    dataframe_val = pd.DataFrame({'value': home_col_list})
                    ema = dataframe_val['value'].ewm(span=min(ema_span, len(home_col_list)), adjust=True).mean().iloc[-1]
                    df.at[index, col + '_P1'] = ema

                    # Update Visitor
                    start, end = self.calcStartEndIdx(visitor_date_index, max_window)
                    visitor_col_list = track_dict[f'{year}_{visitor_team}_{col}'][start:end]
                    dataframe_val = pd.DataFrame({'value': visitor_col_list})
                    ema = dataframe_val['value'].ewm(span=min(ema_span, len(visitor_col_list)), adjust=True).mean().iloc[-1]
                    df.at[index, col + '_P2'] = ema
        
        
            # --------------------------------------------------- 
            # ------------------ Custom Columns -----------------
            # --------------------------------------------------- 
            
            # 1. Add variant of Bill James pythagorean expectation (NFL).
            # 1a. For total points (pythagorean P1, pythagorean P2)
            # Recent games weighted more heavily since 'Final' columns not excluded from the above loop
            start, end = self.calcStartEndIdx(home_date_index, max_window)
            home_points_for = np.array(track_dict[f'{year}_{home_team}_Total'][start:end])
            home_points_against = np.array(track_dict[f'{year}_{home_team}_Total_Allowed'][start:end])
            home_win_pct = track_dict[f'{year}_{home_team}_Win'][start:end]
            optimal_exp = self.get_optimal_exponent(home_points_for, home_points_against, home_win_pct)
            home_points_for = sum(home_points_for)
            home_points_against = sum(home_points_against)
            df.at[index, 'pythagorean_P1'] = home_points_for**optimal_exp / (home_points_for**optimal_exp + home_points_against**optimal_exp)

            start, end = self.calcStartEndIdx(visitor_date_index, max_window)
            visitor_points_for = np.array(track_dict[f'{year}_{visitor_team}_Total'][start:end])
            visitor_points_against = np.array(track_dict[f'{year}_{visitor_team}_Total_Allowed'][start:end])
            visitor_win_pct = np.array(track_dict[f'{year}_{visitor_team}_Win'][start:end])
            optimal_exp = self.get_optimal_exponent(visitor_points_for, visitor_points_against, visitor_win_pct)
            visitor_points_for = sum(visitor_points_for)
            visitor_points_against = sum(visitor_points_against)
            df.at[index, 'pythagorean_P2'] = visitor_points_for**optimal_exp / (visitor_points_for**optimal_exp + visitor_points_against**optimal_exp)
        
            # 2. Add ELO for home, away
            df.at[index, 'elo_P1'] = track_dict[f'{year}_{home_team}_elo'][home_date_index-1]
            df.at[index, 'elo_P2'] = track_dict[f'{year}_{visitor_team}_elo'][visitor_date_index-1]
        
        df.drop(indices_to_drop, inplace=True)
        
        # Add custom metrics to track_cols so it creates the difference (D_) column
        # track_cols.append('pythagorean')
        for col in track_cols[1:]:
            # cont_cols.append('D_' + col)
            df['D_' + col] = (df[col + '_P1'] - df[col + '_P2']).round(3) # Round to 3 sig figs
            
        # Remove columns that begin with H_ or V_ in df
        # y_col = ['H_Won', 'H_start_odds', 'V_start_odds']
        y_col = ['H_start_odds', 'V_start_odds']
        df = df.loc[:, ~(df.columns.str.startswith(('H_', 'V_')) & (~df.columns.isin(y_col)))]
        
        
        df['Over_74'] = df['Over_74'].astype(np.int64)
        df['G5'] = df['G5'].astype(np.int64)
        df['G4'] = df['G4'].astype(np.int64)
        df['G34'] = df['G34'].astype(np.int64)
        df['G45'] = df['G45'].astype(np.int64)
        return df
        # df.to_csv(f'./ttennisData/gen/TTCupSliding{minimum_window}_{max_window}_{ema_span}.csv')


    def get_optimal_exponent(self, x_for, x_against, win_pct):
        """
        For calculating the optimal exponent in pythagorean expecatation
        
        x_for: np.array up to & not including the current game
        x_against: np.array up to & not including the current game
        win_pct: win pct array
        
        with x being home_pts or score

        Maybe use margin based scaling 
        https://chatgpt.com/c/68f81378-7ddc-8326-907c-10ed768953e3
        
        """
        eps = 1e-6
        # Clip win_pct away from 0/1 to avoid degenerate residuals
        win_pct = np.clip(win_pct, eps, 1 - eps)

        window = 5
        win_pct = np.convolve(win_pct, np.ones(window)/window, mode='valid')
        x_for = x_for[-len(win_pct):]
        x_against = x_against[-len(win_pct):]

        # Ensure strictly positive values before taking logs
        logf = np.log(x_for + eps)
        loga = np.log(x_against + eps)
        
        def pythagorean_expectation(inputs, exponent):
            lf, la = inputs
            z = exponent * (la-lf)
            # clip z to avoid exp overflow; exp(700) ~ 1e304, safe upper bound for float64
            z = np.clip(z, -700, 700)
            return 1.0 / (1.0 + np.exp(z))
            
        params, _ = curve_fit(pythagorean_expectation, (logf, loga), win_pct,
                            p0=1.83, bounds=(0.1,10), maxfev=20000)
        return params[0] # optimal exponent


    def calcStartEndIdx(self, date_index, max_window):
        """
        date_index: home_date_index or visitor_date_index. Basically the index of the current record

        RETURNS:
            start_index: The index that goes as far back as max_window (or 0 if < array length) 
            end_index: The index right before the current record
        """
        end_index = date_index - 1
        start_index = max(end_index - max_window, 0)
        return (start_index, end_index)
        
    def makePrediction(self, df, x_col, model_name, exclude_lower_bound, combine_mask=None, return_mask=False, limit=0.05, perf_conts_shift=0.0, metric="74.5"):
        """
        df: copy of datafame

        X_col: List of columns to filter from df

        model_name: name of model in /models/ directory. Ex: 'ttennis' for ttennis_classifier, ttennis_calibrator

        exclude_lower_bound: Set to 1.0 to exclude lower bound. Used for predicting G5

        combine_mask: If defined, this will combine the mask passed in with the one generated

        return_mask: True if you want to return mask and not send any notifications. False if you want to apply and send notifications

        limit: Primary means of defining a lower/upper bounds. limit of 0.05 will exclude ranges inside 0.05 of 0.5.

        perf_conts_shift: used alongside exclude_lower_bound=1.0 for models that are making unconfident predictions, this pushes one side to the other essentially

        metric: What's in the notification. example, over/under <metric>. Example: 74.5

        Generally for adding a new model you'll need a few things:
        returns mask which can be joined with mask = (mask1 == 1) & (mask2 == 1)
            https://stackoverflow.com/questions/15579260/how-to-combine-multiple-numpy-masks
        """
        identifier_col = ['Date', 'Player1_Name', 'Player2_Name', 'Player1', 'Player2']
        # Get upcoming data
        dfUpcoming = df[df['Total_Score'] == 0]

        xId = dfUpcoming[identifier_col]
        x = dfUpcoming[x_col]

        conts = np.stack([x[col].values for col in list(x.columns)], 1)
        idConts = np.stack([xId[col].values for col in list(xId.columns)], 1)

        # Import saved classifier, calibrator (will error out if DNE)
        with open(f'models/ttennis_classifier_{model_name}.pkl', 'rb') as f:
            classifier = pickle.load(f)

        with open(f'models/ttennis_calibrator_{model_name}.pkl', 'rb') as f:
            calibrator = pickle.load(f)

        print(f"Clas: {classifier}")
        print(f"Cal: {calibrator}")
        print(f"conts: {conts.shape}")
        # Apply the model to the upcoming data
        perf_conts_res = classifier.predict_proba(conts)[:,1]
        perf_conts_calibrated = calibrator.predict(perf_conts_res)
        perf_conts_calibrated = perf_conts_calibrated + perf_conts_shift
        perf_conts_calibrated_2d = np.stack([
            1 - perf_conts_calibrated,  # Index 0: P(class=0)
            perf_conts_calibrated       # Index 1: P(class=1)
        ], axis=1)
        perf_conts_calibrated = perf_conts_calibrated_2d
        positive_probs = perf_conts_calibrated[:, 1]

        # Define bounds around 0.5
        lower_bound = 0.5 - limit - exclude_lower_bound
        upper_bound = 0.5 + limit

        # Filter lower/upper bound
        mask = (positive_probs < lower_bound) | (positive_probs > upper_bound)
        # Combine masks if combine_mask is not None
        if combine_mask is not None:
            combined_mask = (mask == 1) & (combine_mask == 1)
            mask = combined_mask

        perf_conts_calibrated = perf_conts_calibrated[mask]
        conts_id = idConts[mask]

        # return mask if return_mask == True
        if return_mask == True:
            return mask

        # ---------------------------------------
        # ---------- Report results -------------
        # ---------------------------------------

        # Date (unix time), notification message
        notifDf = pd.read_csv("/home/ubuntu/ttennis/TTNotif.csv")
        numpyNotifDf = notifDf.to_numpy()

        for i in range(0,len(perf_conts_calibrated)):
            # NOTE: Player_1 is considered 'Home'. When H_won is set to 1, that means it expects player 1 to win.
            # Thus, if index 1 is greater than index 0, it expects player_1 (home) to win. In old terms, left to win not right.
            home = perf_conts_calibrated[i][1] > perf_conts_calibrated[i][0]

            # # Create searchString
            # dateFloat = float(conts_id[i][0])
            # playerOneFloat = float(conts_id[i][3])
            # playerTwoFloat = float(conts_id[i][4])
            # searchString = f"{dateFloat},{playerOneFloat},{playerTwoFloat},{metric}"

            # Get timestamp, convert from UTC+8 to UTC
            # edt_dt = datetime.utcfromtimestamp(int(conts_id[i][0])) - timedelta(hours=4)
            # edt_dt = edt_dt.astimezone(ZoneInfo("America/New_York"))
            # dts = edt_dt.strftime('%Y-%m-%d %H:%M:%S')

            # Assuming conts_id[i][0] is a Unix timestamp in seconds
            utc_dt = datetime.fromtimestamp(int(conts_id[i][0]), tz=timezone.utc)
            edt_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
            dts = edt_dt.strftime('%Y-%m-%d %H:%M:%S')

            homeAway = "home" if home == True else "away"
            # resString = f"Date: {dts}, {homeAway} {metric}, {conts_id[i][1]} vs {conts_id[i][2]}, SearchString: {searchString}\n"
            resString = f"{conts_id[i][1]} vs {conts_id[i][2]}"
            messageString = edt_dt.strftime("%A %-I:%M %p") + f", {homeAway} {metric}"

            edt_dt_plus = edt_dt + timedelta(minutes=4)

            # # Skip if already sent notification, otherwise add to TTNotif.csv
            # row = [ dts, resString ]
            # chkRow = np.array(row)
            # existing_record = np.any(np.all(numpyNotifDf == chkRow, axis=1))
            # if (existing_record):
            #     continue

            # numpyNotifDf = np.append(numpyNotifDf, [row], axis=0)  # Append new row to NumPy array
            # numpyDf = pd.DataFrame(numpyNotifDf, columns=["Date","Notification"]) 
            # numpyDf.to_csv("/home/ubuntu/ttennis/TTNotif.csv", index=False)
            # print(f"Data saved to TTNotif.csv with {len(numpyNotifDf)} records.")

            # data = {
            #     "token": "aad1cbqubtrb1vhfdcwahxfbwgyhmy",
            #     "user": "u7wrjudz5tkags2ncr862u591pg8bd",
            #     "message": messageString,
            #     "title": resString
            # }
            # response = requests.post("https://api.pushover.net/1/messages.json", data=data)
            # print(response.status_code, response.text)

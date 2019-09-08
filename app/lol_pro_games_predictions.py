#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle


class ProGamesPredictions:

    def __init__(self):
        self.data = self.get_data()
        self.model = self.get_model()
        self.teams_in_data = np.unique(self.data['team'])

    @staticmethod
    def get_data():
        data = pd.read_csv('data/lol_pro_games_data_0407.csv')

        return data

    @staticmethod
    def get_model():
        with open('data/model_0407_01.pkl', 'rb') as model_pkl:
            lgb_clf = pickle.load(model_pkl)

        return lgb_clf

    @staticmethod
    def swap_pairs(pair):
        all_pairs = []
        all_pairs.append(pair)
        all_pairs.append([pair[1], pair[0]])

        return all_pairs

    def predict(self, team_1, team_2):
        result = {
            'team_1': team_1,
            'team_2': team_2,
        }

        if self.check_correct(team_1, team_2):
            pairs = [team_1, team_2]
            pairs = self.swap_pairs(pairs)
            features_df = self.get_features_by_pairs(self.data, pairs)

            info_columns = ['team', 'rival']
            X_test = features_df.drop(info_columns, axis=1)
            preds = self.model.predict(X_test)

            for ind, _ in enumerate(preds):
                if ind % 2 == 0:
                    argmax_prediction = np.argmax(preds[ind:ind + 2])
                    if argmax_prediction == 0:
                        result['prediction'] = preds[ind]
                    elif argmax_prediction == 1:
                        result['prediction'] = 1 - preds[ind+1]

            #for (team_1, team_2), prediction in zip(pairs, preds):
                #print('{0} VS {1} prediction {2:.3}'.format(team_1, team_2, prediction))

        else:
            result['prediction'] = np.nan

        return result

    def check_correct(self, team_1, team_2):
        if team_1 == team_2:
            raise ValueError('сначала подумай, потом запрашивай предикт')
        elif (team_1 or team_2) not in self.teams_in_data:
            raise ValueError('ну блин, данных нет, сорян, запроси другие команды')
        else:
            return True

    def get_features_by_pairs(self, df, pairs):
        teams = np.unique(pairs)
        pairwise_winrates = self.calc_pairwise_winrate(df, pairs)
        mean_features_by_teams = self.calc_mean_features_by_teams(df, teams)
        pairwise_mean_features = self.calc_pairwise_mean_features(df, pairs)

        full_df = pd.DataFrame()
        for ind, (team_1, team_2) in enumerate(pairs):
            tmp_1 = mean_features_by_teams[mean_features_by_teams['team'] == team_1]
            tmp_2 = mean_features_by_teams[mean_features_by_teams['team'] == team_2].rename(columns={'team': 'rival'})
            pair_tmp = pairwise_winrates.merge(tmp_1, on=['team'])
            pair_tmp = pair_tmp.merge(tmp_2, on=['rival'])

            full_df = full_df.append(pair_tmp)

        full_df = full_df.merge(pairwise_mean_features, on=['team', 'rival'])

        return full_df

    def calc_pairwise_winrate(self, matches_data, pairs):
        pairwise_winrates = []
        matches_results = matches_data.loc[matches_data['position'] == 'Team', ['gameid', 'team', 'rival', 'result']]
        
        for team_1, team_2 in pairs:
            tmp = matches_results[(matches_results['team'] == team_1) & (matches_results['rival'] == team_2)]

            if len(tmp) == 0:
                winrate = np.nan
            else:
                winrate = tmp['result'].mean()

            pairwise_winrates.append({
                'team': team_1,
                'rival': team_2,
                'pairwise_winrate': winrate
            })

        pairwise_winrates = pd.DataFrame(pairwise_winrates)
        return pairwise_winrates[['team', 'rival', 'pairwise_winrate']]

    def calc_mean_features_by_teams(self, matches_data, teams):
        mean_features = []
        names_of_features = [
            'kpm', 'okpm', 'teamtowerkills', 'opptowerkills',
            'dmgtochampsperminute', 'wpm', 'wcpm', 'cspm',
            'minionkills', 'monsterkillsownjungle', 'monsterkillsenemyjungle', 'result',
        ]
        teams_data = matches_data[matches_data['position'] == 'Team']

        for team in teams:
            tmp = teams_data[teams_data['team'] == team]

            tmp_agg = {}
            if len(tmp) == 0:
                for key in names_of_features:
                    tmp_agg[key] = None
            else:
                for key in names_of_features:
                    tmp_agg[key] = np.nanmean(tmp[key])

            tmp_agg['team'] = team
            mean_features.append(tmp_agg)

        mean_features = pd.DataFrame(mean_features)
        mean_features.rename(columns={'result': 'winrate'}, inplace=True)
        return mean_features

    def calc_pairwise_mean_features(self, matches_data, pairs):
        mean_features = []
        names_of_features = [
            'dmgtochampsperminute', 'wpm', 'wcpm',
            'earnedgpm', 'minionkills', 'monsterkills',
            'monsterkillsownjungle', 'monsterkillsenemyjungle',
        ]
        players_data = matches_data[matches_data['position'] != 'Team']
        unique_teams = np.unique(pairs)
        unique_lanes = np.unique(players_data.position.values)

        for team in unique_teams:
            tmp = players_data[players_data['team'] == team]

            tmp_agg = {}
            if len(tmp) == 0:
                for lane in unique_lanes:
                    for key in names_of_features:
                        tmp_agg[key + '_' + lane.lower()] = None
            else:
                for lane in unique_lanes:
                    tmp_by_lane = tmp[tmp['position'] == lane]
                    for key in names_of_features:
                        tmp_agg[key + '_' + lane.lower()] = np.nanmean(tmp_by_lane[key])

            tmp_agg['team'] = team
            mean_features.append(tmp_agg)

        mean_features = pd.DataFrame(mean_features)
        features = list(set(mean_features.columns) - set(['team']))

        pairwise_statistics = []
        for team_1, team_2 in pairs:
            team_1_data = mean_features[mean_features['team'] == team_1]
            team_2_data = mean_features[mean_features['team'] == team_2]

            statistics_by_features = {}
            for f in features:
                statistics_by_features[f + '_pairwise'] = team_1_data[f].values[0] / (team_2_data[f].values[0] + 1)

            statistics_by_features['team'] = team_1
            statistics_by_features['rival'] = team_2
            pairwise_statistics.append(statistics_by_features)

        pairwise_statistics = pd.DataFrame(pairwise_statistics)

        return pairwise_statistics


def test0(team_1='FlyQuest', team_2='Echo Fox'):
    model = ProGamesPredictions()
    predictions = model.predict(team_1, team_2)
    print('{0} VS {1}: {2}'.format(
        predictions['team_1'],
        predictions['team_2'],
        predictions['prediction']
    ))


#if __name__ == '__main__':
#    pairs_for_test =  [
#        ['FlyQuest', 'Echo Fox'],
#        ['Team Liquid', 'Cloud9'],
#        ['Team SoloMid', '100 Thieves'],
#        ['Golden Guardians', 'OpTic Gaming'],
#        ['Clutch Gaming', 'Counter Logic Gaming']
#    ]
#    for pair in pairs_for_test:
#        test0(pair[0], pair[1])

# Fiche résultats — projet Détection de fraude

> Générée automatiquement par `python -m src.report` depuis `reports/results.json` (modèle `1.0.0+20261022T0005Z`). Ne pas modifier à la main : relancer `make train` (ou `python -m src.report`).

## 1. Cases jaunes du dossier

| Case | Valeur |
|---|---|
| PR-AUC test, XGBoost calibré | 0,777 [IC 95 % : 0,692–0,850] |
| Réduction du coût, seuil t* vs seuil 0,5 (modèle non calibré, `predict()`) | 16,4 % (IC 95 % : -11,6 % à 42,8 %) |
| Réduction du coût, seuil t* vs seuil 0,5 (probabilités calibrées) | 14,1 % (IC 95 % : -12,3 % à 41,0 %) |
| Réduction du coût, top-K rentable vs seuil 0,5 (non calibré) | 37,1 % (IC 95 % : 12,4 % à 59,1 %) |
| Réduction du coût, top-K rentable vs seuil t* | 24,7 % (IC 95 % : 4,3 % à 48,4 %) |
| Brier score test avant → après calibration | 0,000714 → 0,000500 |
| Calibration retenue | Platt (sigmoïde) |
| Seuil optimal t* | 0,856 (seuil F1-optimal : 0,658) |
| Capacité K retenue | 10 alertes/heure |
| Top-3 features SHAP | V14, V4, V12 |
| Montant médian fraude vs légitime | 9,25 € vs 22,00 € → **plus faible** |
| Taux de fraude nuit (0h–6h) vs jour | 0,518 % vs 0,141 % → **plus élevé la nuit** |
| Composantes les plus séparatrices (stat. KS) | V14 (0,84), V10 (0,80), V12 (0,78), V4 (0,77) |

## 2. Tableau de résultats (section 8 du dossier)

Test = dernière période (38,7 h → 48,0 h, 71 202 transactions, 94 fraudes, 10 839 € de fraude). Toutes les politiques sont évaluées **à capacité constante K = 10/heure**. Coûts : c_inv = 5 €, c_fp = 10 €, r = 0,80.

| Modèle / politique | PR-AUC test [IC 95 %] | Recall | Precision | Coût total |
|---|---|---|---|---|
| Dummy (seuil 0,5) | 0,001 [IC 95 % : 0,001–0,002] | 0,0 % | 0,0 % | 10 839 € |
| Régression logistique (seuil 0,5) | 0,684 [IC 95 % : 0,581–0,786] | 2,1 % | 2,0 % | 12 256 € |
| Random Forest (seuil 0,5) | 0,807 [IC 95 % : 0,728–0,878] | 62,8 % | 93,7 % | 6 223 € |
| XGBoost non calibré, seuil 0,5 | 0,777 [IC 95 % : 0,692–0,850] | 51,1 % | 58,5 % | 7 412 € |
| XGBoost calibré, seuil 0,5 | 0,777 [IC 95 % : 0,692–0,850] | 59,6 % | 78,9 % | 7 213 € |
| XGBoost, seuil F1-optimal | 0,777 [IC 95 % : 0,692–0,850] | 59,6 % | 83,6 % | 6 359 € |
| XGBoost, seuil coût-optimal t* | 0,777 [IC 95 % : 0,692–0,850] | 48,9 % | 100,0 % | 6 197 € |
| XGBoost, top-K perte espérée | 0,777 [IC 95 % : 0,692–0,850] | 39,4 % | 37,0 % | 5 253 € |
| XGBoost, top-K perte espérée (rentables seulement) | 0,777 [IC 95 % : 0,692–0,850] | 28,7 % | 77,1 % | 4 665 € |
| *Référence : aucune alerte* | — | 0 % | — | 10 839 € |

## 3. Phrases à dire (avec tes vrais chiffres)

**Résultat STAR (1.2).** « À capacité d'analystes constante (10 alertes/heure), la priorisation par perte espérée (chaque heure, les K transactions de plus forte perte p × montant, investigations rentables seulement) réduit le coût total de 37,1 % par rapport au seuil standard 0,5 du modèle, avec une PR-AUC test de 0,777 [IC 95 % : 0,692–0,850]. » À savoir : IC 95 % : 12,4 % à 59,1 %, gain significatif. Pour le seuil coût-optimal (t* = 0,856) : réduit le coût total de 16,4 % (IC 95 % : -11,6 % à 42,8 % : l'intervalle contient 0, ce n'est pas significatif sur 94 fraudes).

**Sensibilité (5.3).** « Si le coût client d'un faux positif double (10 € → 20 €), le seuil optimal ne bouge pas (t* = 0,856) : avec K = 10, c'est l'heure la plus chargée de la validation qui fixe le seuil : t* est le seuil le moins strict qui ne dépasse jamais 10 alertes dans une heure. Il est identique pour les 16 hypothèses de coûts testées ; seule la capacité le fait bouger : K = 5 → t* = 0,861, coût test 7 476 € ; K = 10 → t* = 0,856, coût test 6 197 € ; K = 20 → t* = 0,840, coût test 6 214 € ; K = 40 → t* = 0,658, coût test 5 110 €. Le levier de décision, c'est la capacité de l'équipe, pas le coût client. »

**Q4 — SMOTE ou poids de classe ?** Sur ce dataset (validation), PR-AUC : pondération 0,807 [0,741–0,868], SMOTE 0,798 [0,732–0,861], sans rééquilibrage 0,799. Le meilleur point est la pondération, mais les intervalles se recouvrent : la différence n'est pas significative. J'ai gardé la pondération : plus simple, pas de données synthétiques, et la calibration corrige ensuite les probabilités. Brier (non calibré) : pondération 0,00078, SMOTE 0,00130, sans rééquilibrage 0,00062.

**Q5 — Pourquoi XGBoost ?** PR-AUC en validation : XGBoost 0,807 [0,741–0,868] contre Random Forest 0,774 [0,694–0,856] ; sur le test : 0,777 [0,692–0,850] contre 0,807 [0,728–0,878]. Les intervalles se recouvrent : sur ce dataset, XGBoost n'apporte pas de gain significatif sur la Random Forest, et je le dis. Je l'ai gardé comme modèle principal (choix fait avant de regarder le test) pour l'industrialisation : modèle compact (300 arbres de profondeur 3) et TreeSHAP natif pour les reason codes de l'API.

**Q9 — Isotonic ou Platt ?** J'ai retenu Platt (sigmoïde). Les deux méthodes sont à égalité sur le Brier (Brier hors-échantillon, cross-fitting en 2 blocs chronologiques sur la validation) : 0,000567 contre 0,000566 (écart < 1 %), et la log-loss les départage nettement : 0,0035 contre 0,0058 : isotonic donne une probabilité < 0,0001 à 4 fraudes contre 0, or la décision utilise p × montant. C'est cohérent avec la théorie : Platt est plus stable avec peu de positifs (129 fraudes en validation). Sans calibration : 0,000781. Sur le test, le Brier passe de 0,000714 à 0,000500.

**Q12 — Seuil ou top-K ?** Dans mes résultats, le top-K par perte espérée (investigations rentables seulement) coûte 24,7 % de moins que le seuil coût-optimal (4 665 € contre 6 197 € sur le test ; IC 95 % : 4,3 % à 48,4 %, gain significatif). Ce que montrent les chiffres : en classant par p × montant, il intercepte 74 % du montant de fraude en détectant 27 fraudes, contre 56 % avec 46 fraudes pour le seuil, qui ignore le montant : il vise les grosses fraudes plutôt que les plus probables, et ne gaspille pas d'investigation sur les petits montants. Je recommanderais le top-K pour la file des analystes, et garderais le seuil t* comme règle temps réel simple à auditer. Le top-K « pur » (toujours 10 alertes/heure, même non rentables) coûte 5 253 €.

**Q24 — Pour un directeur financier.** « À capacité d'analystes constante, la priorisation des alertes par perte espérée réduit le coût total de la fraude de 37,1 % par rapport au seuil standard, tout en diminuant les blocages de clients légitimes de 76,5 % (34 → 8 faux positifs). » (IC 95 % : 12,4 % à 59,1 %, gain significatif.)

## 4. Faits à connaître sur la méthode

- **Split temporel** : train 269 fraudes, validation 129, test 94. Le split 60/20/20 ne laissait pas assez de fraudes en validation/test (le taux de fraude baisse le 2e jour), d'où le repli sur 50/25/25.
- **Tuning** : 50 essais Optuna (TPE, 0 élagués), PR-AUC CV temporelle 0,826 (paramètres par défaut : 0,805). Meilleurs paramètres : `max_depth=3`, `learning_rate=0.0925`, `n_estimators=300`, `subsample=0.808`, `colsample_bytree=0.466`, `min_child_weight=7.86`, `reg_lambda=8.66`, `scale_pos_weight=4.64`.
- **scale_pos_weight** : Optuna a retenu 4,6 alors que n_neg/n_pos vaut 528 sur le train (point de départ de la recherche, PR-AUC CV 0,805). Un poids extrême n'est pas optimal en PR-AUC ; la calibration corrige de toute façon les probabilités.
- **Pourquoi t* = 0,856** : sans contrainte, le coût minimal en validation serait à t = 0,812, mais avec jusqu'à 31 alertes dans une heure. t* est le seuil le moins strict qui ne dépasse jamais K = 10 alertes dans aucune heure de validation.
- **Créneaux d'investigation utilisés sur le test** (10 × 10 heures) : XGBoost, seuil coût-optimal t* 46 % ; XGBoost, top-K perte espérée 100 % ; XGBoost, top-K perte espérée (rentables seulement) 35 %.
- **Accuracy trompeuse** : prédire « jamais fraude » donne 99,83 % d'accuracy.
- **Politique t* sur le test** : 46 alertes investiguées (46 fraudes, 0 faux positifs), 56,2 % du montant de fraude intercepté, 0,00 faux positifs pour 1 000 transactions, 2 alertes au-delà de la capacité (max 12/heure).
- **Seuil 0,5 non calibré** : 113 alertes levées dont 31 non traitables faute de capacité.
- **Monitoring** : PSI du score train → test = 0,179 (watch). Features hors zone stable : hour_sin (2,36), hour_cos (2,19), V1 (1,57), V3 (1,52), V28 (1,02), is_night (0,59).
- **SHAP — Vrai positif (fraude alertée par t*)** : montant 667,55 €, p = 0,878 ; reason codes : V14 (+3,33), V12 (+2,33), V17 (+2,25). Figure : `reports/figures/shap_waterfall_true_positive.png`.
- **SHAP — Transaction légitime la plus suspecte (t* n'a aucun faux positif sur le test)** : montant 302,65 €, p = 0,822 ; reason codes : V12 (+2,66), V17 (+2,42), V4 (+1,44). Figure : `reports/figures/shap_waterfall_false_positive.png`.
- **SHAP — Faux négatif le plus coûteux (non alerté par t*)** : montant 1 096,99 €, p = 0,000 ; reason codes : hour_sin (+0,70), log_amount (+0,63), V8 (+0,54). Ce qui l'a fait passer pour légitime : V14 (-1,57), V12 (-0,79). Figure : `reports/figures/shap_waterfall_false_negative.png`.

## 5. Q23 — matière pour « ce que je ferais différemment »

- Seulement 129 fraudes en validation et 94 en test : les IC sont larges, le seuil est choisi sur peu de positifs.
- Le taux de fraude varie dans le temps (train 0,189 %, validation 0,181 %, test 0,132 %) : recalibrer régulièrement est indispensable.
- Les coûts sont des hypothèses : à valider avec l'équipe fraude et la finance (voir l'analyse de sensibilité).
- Pas d'identifiant client : pas de features comportementales (fréquence, montant habituel).

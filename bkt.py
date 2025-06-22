import pandas as pd

class BKTModel:
    pg = 0.2
    ps = 0.3
    pt = 0.1

    def __init__(self):
        pass

    def predict(self, X: pd.DataFrame):
        student_pl = X['PL'].iloc[0]
        student_response = X['correct'].iloc[0]
        
        if student_response == 1:
            pl_after_response = (student_pl * (1 - self.ps)) / (student_pl * (1 - self.ps) + (1 - student_pl) * self.pg)
        else:
            pl_after_response = (student_pl * self.ps) / (student_pl * self.ps + (1 - student_pl) * (1 - self.pg))
        
        new_pl = round(pl_after_response + ((1 - pl_after_response) * self.pt), 2)
        return [new_pl]

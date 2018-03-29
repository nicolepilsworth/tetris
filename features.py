import numpy as np
import pandas as pd

def sameFeatures(a, b_list):
    for b in b_list:
        if np.equal(a.f_list, b.f_list).all():
            return True
    return False

def distinct(seq):
   result = []
   for item in seq:
       if sameFeatures(item, result): continue
       result.append(item)
   return result

def pareto(moves, pareto_type):
    dominance = {
      "simple": simpleDominance,
      "cumulative": cumulativeDominance
    }[pareto_type]

    filtered = []
    for obj in moves:
        include = True
        for obj_compare in moves:
            if dominance(obj, obj_compare):
                include = False
                break
        if include:
            filtered.append(obj)

    return filtered

def simpleDominance(a, b):
    ''' Returns True if a is dominated by b'''
    return np.greater_equal(a.f_list, b.f_list).all() and np.greater(a.f_list, b.f_list).any()

def cumulativeDominance(a, b):
    a_cumulative = np.cumsum(a.f_list)
    b_cumulative = np.cumsum(b.f_list)

    return np.greater_equal(a_cumulative, b_cumulative).all() and np.greater(a_cumulative, b_cumulative).any()

class Features:
    def __init__(self, b, pos, nrows, ncols, eroded, l_height):
        self.b = b
        self.pos = pos
        self.eroded_piece_cells = eroded
        self.landing_height = l_height
        self.nrows = nrows
        self.ncols = ncols
        self.calcFeatures()
        # self.rows_with_holes = 0
        # self.column_transitions = self.b.ncols
        # self.holes = 0
        # self.landing_height = 0
        # self.cum_wells = 0
        # self.row_transitions = 2*self.b.nrows
        # self.eroded_piece_cells = 0
        # self.hole_depth = 0

    def listFeatures(self):
        print(  """Board Features:
                Rows with Holes: {rows_w_holes}
                Column Transitions: {col_trans}
                Holes: {holes}
                Landing Height: {l_height}
                Cumulative Wells: {cum_wells}
                Row Transitions: {row_trans}
                Eroded Piece Cells: {erod_cells}
                Hole Depth: {h_depth}""".format(
                    rows_w_holes=self.rows_with_holes,
                    col_trans=self.column_transitions,
                    holes=self.holes,
                    l_height=self.landing_height,
                    cum_wells=self.cum_wells,
                    row_trans=self.row_transitions,
                    erod_cells=self.eroded_piece_cells,
                    h_depth=self.hole_depth,
                ))

    def calcFeatures(self):

        # Initialisations (reset of feature values)
        holes = 0
        row_transitions = 0
        rows_with_holes = np.zeros(self.nrows)
        cum_wells = 0
        column_transitions = 0
        hole_depth = 0

        hole_depth_helper = 0
        cum_well_helper = 0
        row_trans_helper = np.ones(self.nrows)

        for col in range(self.ncols):
            cum_well_helper = 0
            hole_depth_helper = 0
            # col_state used for column transitions;
            # compare next row up's cell with col_state value.
            # If different, column transition is present. Switch col_state value,
            # continue up rows in column.
            col_state = False
            has_cell_above = False
            for row in reversed(range(self.nrows)):

                if self.b[row][col]: # If any full cell covers empty cell, empty cells are holes
                    has_cell_above = True
                    hole_depth_helper += 1
                    cum_well_helper = 0
                else:
                    if self.isWell(row, col):
                        cum_well_helper += 1
                        cum_wells += cum_well_helper
                    if has_cell_above:
                        holes += 1
                        rows_with_holes[row] = 1
                        hole_depth += hole_depth_helper
                        hole_depth_helper = 0
                if self.b[row][col] != col_state:
                    col_state = not col_state
                    column_transitions += 1

                if self.b[row][col] != row_trans_helper[row]:
                    row_trans_helper[row] = not row_trans_helper[row]
                    row_transitions += 1

            # Final row transition (to 'empty' top row)
            if not col_state:
                column_transitions += 1
        for row in range(self.nrows):
            if not self.b[row][-1]:
              row_transitions += 1

        rows_with_holes = sum(rows_with_holes)

        self.f_list = [
            -1 * self.eroded_piece_cells,
            hole_depth,
            row_transitions,
            cum_wells,
            self.landing_height,
            holes,
            column_transitions,
            rows_with_holes
        ]

    def isWell(self, row, col):
      if col == 0:
        return self.b[row][1]
      elif col == self.ncols - 1:
        return self.b[row][-2]
      else:
        return self.b[row][col-1] and self.b[row][col+1]

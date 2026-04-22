#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py  –  Application-wide constants for CelleditPy.
"""

from PyQt6.QtGui import QColor

VERSION = "0.4.0"

# CPK colour palette (element symbol -> QColor).
# Falls back to DEFAULT (hot pink) for unknown elements.
CPK_COLORS = {
    'H':  QColor('#FFFFFF'), 'C':  QColor('#222222'), 'N':  QColor('#3377FF'),
    'O':  QColor('#FF3333'), 'F':  QColor('#99E6E6'), 'Cl': QColor('#33FF33'),
    'Br': QColor('#A52A2A'), 'I':  QColor('#9400D3'), 'S':  QColor('#FFC000'),
    'P':  QColor('#FF8000'), 'Si': QColor('#DAA520'), 'B':  QColor('#FA8072'),
    'He': QColor('#D9FFFF'), 'Ne': QColor('#B3E3F5'), 'Ar': QColor('#80D1E3'),
    'Kr': QColor('#5CACC8'), 'Xe': QColor('#429EB0'), 'Rn': QColor('#298FA2'),
    'Li': QColor('#CC80FF'), 'Na': QColor('#AB5CF2'), 'K':  QColor('#8F44D7'),
    'Rb': QColor('#702EBC'), 'Cs': QColor('#561B9E'), 'Fr': QColor('#421384'),
    'Be': QColor('#C2FF00'), 'Mg': QColor('#8AFF00'), 'Ca': QColor('#3DFF00'),
    'Sr': QColor('#00FF00'), 'Ba': QColor('#00E600'), 'Ra': QColor('#00B800'),
    'Sc': QColor('#E6E6E6'), 'Ti': QColor('#BFC2C7'), 'V':  QColor('#A6A6AB'),
    'Cr': QColor('#8A99C7'), 'Mn': QColor('#9C7AC7'), 'Fe': QColor('#E06633'),
    'Co': QColor('#F090A0'), 'Ni': QColor('#50D050'), 'Cu': QColor('#C88033'),
    'Zn': QColor('#7D80B0'), 'Ga': QColor('#C28F8F'), 'Ge': QColor('#668F8F'),
    'As': QColor('#BD80E3'), 'Se': QColor('#FFA100'), 'Tc': QColor('#3B9E9E'),
    'Ru': QColor('#248F8F'), 'Rh': QColor('#0A7D8F'), 'Pd': QColor('#006985'),
    'Ag': QColor('#C0C0C0'), 'Cd': QColor('#FFD700'), 'In': QColor('#A67573'),
    'Sn': QColor('#668080'), 'Sb': QColor('#9E63B5'), 'Te': QColor('#D47A00'),
    'La': QColor('#70D4FF'), 'Ce': QColor('#FFFFC7'), 'Pr': QColor('#D9FFC7'),
    'Nd': QColor('#C7FFC7'), 'Pm': QColor('#A3FFC7'), 'Sm': QColor('#8FFFC7'),
    'Eu': QColor('#61FFC7'), 'Gd': QColor('#45FFC7'), 'Tb': QColor('#30FFC7'),
    'Dy': QColor('#1FFFC7'), 'Ho': QColor('#00FF9C'), 'Er': QColor('#00E675'),
    'Tm': QColor('#00D452'), 'Yb': QColor('#00BF38'), 'Lu': QColor('#00AB24'),
    'Hf': QColor('#4DC2FF'), 'Ta': QColor('#4DA6FF'), 'W':  QColor('#2194D6'),
    'Re': QColor('#267DAB'), 'Os': QColor('#266696'), 'Ir': QColor('#175487'),
    'Pt': QColor('#D0D0E0'), 'Au': QColor('#FFD123'), 'Hg': QColor('#B8B8D0'),
    'Tl': QColor('#A6544D'), 'Pb': QColor('#575961'), 'Bi': QColor('#9E4FB5'),
    'Po': QColor('#AB5C00'), 'At': QColor('#754F45'), 'Ac': QColor('#70ABFA'),
    'Th': QColor('#00BAFF'), 'Pa': QColor('#00A1FF'), 'U':  QColor('#008FFF'),
    'Np': QColor('#0080FF'), 'Pu': QColor('#006BFF'), 'Am': QColor('#545CF2'),
    'Cm': QColor('#785CE3'), 'Bk': QColor('#8A4FE3'), 'Cf': QColor('#A136D4'),
    'Es': QColor('#B31FD4'), 'Fm': QColor('#B31FBA'), 'Md': QColor('#B30DA6'),
    'No': QColor('#BD0D87'), 'Lr': QColor('#C70066'), 'Al': QColor('#B3A68F'),
    'Y':  QColor('#99FFFF'), 'Zr': QColor('#7EE7E7'), 'Nb': QColor('#68CFCE'),
    'Mo': QColor('#52B7B7'),
    'DEFAULT': QColor('#FF1493'),  # hot-pink fallback
}

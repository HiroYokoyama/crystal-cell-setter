#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py  –  Application-wide constants for CelleditPy.
"""

VERSION = "0.4.1"

# CPK colour palette (element symbol -> hex string).
# Falls back to DEFAULT (hot pink) for unknown elements.
# Stored as plain hex strings so this module has no Qt dependency
# and can be safely imported in headless / test environments.
CPK_COLORS: dict[str, str] = {
    'H':  '#FFFFFF', 'C':  '#222222', 'N':  '#3377FF',
    'O':  '#FF3333', 'F':  '#99E6E6', 'Cl': '#33FF33',
    'Br': '#A52A2A', 'I':  '#9400D3', 'S':  '#FFC000',
    'P':  '#FF8000', 'Si': '#DAA520', 'B':  '#FA8072',
    'He': '#D9FFFF', 'Ne': '#B3E3F5', 'Ar': '#80D1E3',
    'Kr': '#5CACC8', 'Xe': '#429EB0', 'Rn': '#298FA2',
    'Li': '#CC80FF', 'Na': '#AB5CF2', 'K':  '#8F44D7',
    'Rb': '#702EBC', 'Cs': '#561B9E', 'Fr': '#421384',
    'Be': '#C2FF00', 'Mg': '#8AFF00', 'Ca': '#3DFF00',
    'Sr': '#00FF00', 'Ba': '#00E600', 'Ra': '#00B800',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V':  '#A6A6AB',
    'Cr': '#8A99C7', 'Mn': '#9C7AC7', 'Fe': '#E06633',
    'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033',
    'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
    'As': '#BD80E3', 'Se': '#FFA100', 'Tc': '#3B9E9E',
    'Ru': '#248F8F', 'Rh': '#0A7D8F', 'Pd': '#006985',
    'Ag': '#C0C0C0', 'Cd': '#FFD700', 'In': '#A67573',
    'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00',
    'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7',
    'Nd': '#C7FFC7', 'Pm': '#A3FFC7', 'Sm': '#8FFFC7',
    'Eu': '#61FFC7', 'Gd': '#45FFC7', 'Tb': '#30FFC7',
    'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675',
    'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24',
    'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W':  '#2194D6',
    'Re': '#267DAB', 'Os': '#266696', 'Ir': '#175487',
    'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
    'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5',
    'Po': '#AB5C00', 'At': '#754F45', 'Ac': '#70ABFA',
    'Th': '#00BAFF', 'Pa': '#00A1FF', 'U':  '#008FFF',
    'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2',
    'Cm': '#785CE3', 'Bk': '#8A4FE3', 'Cf': '#A136D4',
    'Es': '#B31FD4', 'Fm': '#B31FBA', 'Md': '#B30DA6',
    'No': '#BD0D87', 'Lr': '#C70066', 'Al': '#B3A68F',
    'Y':  '#99FFFF', 'Zr': '#7EE7E7', 'Nb': '#68CFCE',
    'Mo': '#52B7B7',
    'DEFAULT': '#FF1493',  # hot-pink fallback
}

#!/usr/bin/env python
# coding: utf-8

# In[25]:


def select_grid(grid):
    switcher = {
        'cams': '/grid/grid0_1.gpkg',
        's5p': '/grid/grid0_066.gpkg',
        'arpa': '/grid/grid0_01.gpkg'
    }
    return switcher.get(grid, "Invalid year grid")

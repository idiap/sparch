#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-clause
#
# This file is part of the sparch package
#

from distutils.core import setup

import setuptools

setup(
    name="sparch",
    version="1.0",
    description="Spiking Neural Architectures for Speech Technology",
    author="Alexandre Bittar",
    author_email="abittar@idiap.ch",
    packages=setuptools.find_packages(),
)

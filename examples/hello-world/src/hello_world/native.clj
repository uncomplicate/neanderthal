;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.native
  (:require [uncomplicate.neanderthal
             [core :refer :all]
             [native :refer :all]]))

;; We create two matrices...
(def a (dge 2 3 [1 2 3 4 5 6]))
(def b (dge 3 2 [1 3 5 7 9 11]))
;; ... and multiply them
(mm a b)

;; If you see something like this:
;; #RealGEMatrix[double, mxn:2x2, layout:column, offset:0]
;; ▥       ↓       ↓       ┓
;; →       35.0    89.0
;; →       44.0   116.0
;; ┗                       ┛
;; It means that everything is set up and you can enjoy programming with Neanderthal :)

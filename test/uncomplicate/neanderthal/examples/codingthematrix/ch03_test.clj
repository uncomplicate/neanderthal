;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.examples.codingthematrix.ch03-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [axpy]]
             [native :refer [dv dge]]]))

(facts
 "3.1 Linear combination"
 (axpy -5 (dv 2 3.5) 2 (dv 4 10)) => (dv -2 2.5)

 (let [garden-gnome (dv 0 1.3 0.2 0.8 0.4)
       hula-hoop (dv 0 0 1.5 0.4 0.3)
       slinky (dv 0.25 0 0 0.2 0.7)
       silly-putty (dv 0 0 0.3 0.7 0.5)
       salad-shooter (dv 0.15 0 0.5 0.4 0.8)]
   (axpy 240 garden-gnome 55 hula-hoop 150 slinky
         133 silly-putty 90 salad-shooter)
   => (dv 51 312 215.4 373.09999999999997 356)))

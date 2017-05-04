(ns uncomplicate.neanderthal.cublas-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [cuda :refer [with-engine *cuda-factory* cuda-handle cuda-float cuda-double]]
             [block-test :as block-tes]
             [real-test :refer :all]
             ]))

(defn test-blas1 [factory]
  ;;(test-group factory)
  ;;(test-vctr-constructor factory)
  (test-vctr factory)
  (test-vctr-bulk-entry! factory)
  (test-dot factory)
  (test-sum factory)
  (test-iamax factory)
  (test-vctr-nrm2 factory)
  (test-vctr-asum factory)
  (test-vctr-swap factory)
  (test-vctr-copy factory)
  (test-vctr-scal factory)
  (test-vctr-axpy factory)
  ;;(test-ge-constructor factory)
  ;;(test-ge factory)
  ;;(test-ge-bulk-entry! factory)
  ;;(test-ge-swap factory)
  ;;(test-ge-copy factory)
  ;;(test-ge-scal factory)
  ;;(test-ge-axpy factory)
  ;;(test-ge-mv factory)
  ;;(test-rk factory)
  ;;(test-ge-mm factory)
  ;;(test-tr factory)
  ;;(test-tr-constructor factory)
  ;;(test-tr-copy factory)
  ;;(test-tr-swap factory)
  ;;(test-tr-scal factory)
  ;;(test-tr-axpy factory)
  ;;(test-tr-mv factory)
  ;;(test-tr-mm factory)
  )

(with-release [handle (cuda-handle)]

  (with-engine cuda-float handle
;;    (block-test/test-all *cuda-factory*)
    (test-blas1 *cuda-factory*))

  (with-engine cuda-double handle
;;    (block-test/test-all *cuda-factory*)
    (test-blas1 *cuda-factory*)))

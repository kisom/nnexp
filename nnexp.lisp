;;;; nnexp.lisp

(in-package #:nnexp)

;;; "nnexp" goes here. Hacks and glory await!
;;;
;;; The basic neural network should model a three-layer perceptron
;;; network; right now, online backpropagation is being used for
;;; training.

(defclass layer ()
  ((bias :type 'foreign-array :initarg :bias :accessor bias-of)
   (weights :type 'foreign-array :initarg :weights :accessor weights-of))
  (:documentation "A layer contains the weights and bias for a layer in the neural network."))

(defun weight-random (limit)
  (/ (random limit) (random limit)))

(defun random-1 (n &key (limit 1.0) (rand #'random))
  "Create a random n-by-1 matrix."
  (let ((tmplst nil))
    (dotimes (i n)
      (push (make-array 1 :element-type 'single-float
			:initial-contents (list (funcall rand limit))) tmplst))
    (setf tmplst (make-array (list n 1) :initial-contents tmplst))
    (lisp-matrix:make-matrix n 1
			     :implementation :foreign-array
			     :element-type 'lisp-matrix:single-float
			     :initial-contents tmplst)))

(defun random-m-n (m n &key (limit 1.0) (rand #'random))
  (let ((tmparr nil)
	(tmprow nil))
    (dotimes (j m)
      (setf tmprow nil)
      (dotimes (i n)
	(push (funcall rand limit) tmprow))
      (setf tmprow (make-array n :initial-contents tmprow))
      (push tmprow tmparr))
    (setf tmparr (make-array (list m n) :initial-contents tmparr))
    (lisp-matrix:make-matrix m n
			     :implementation :foreign-array
			     :element-type 'lisp-matrix:single-float
			     :initial-contents tmparr)))

(defun make-layer (inputs neurons &key (limit 1.0))
  (make-instance 'layer
		 :weights (random-m-n inputs neurons :limit limit :rand #'weight-random)
		 :bias    (random-1   neurons :limit limit)))

(defun make-network-3 (inputs hidden outputs &key (limit 1.0))
  "Create a 3-layer network."
  (acons :layers
	 (list (make-layer inputs hidden  :limit limit)
	       (make-layer hidden outputs :limit limit))
	 (acons :activation :sigmoid nil)))

(defun input-matrix (inputs)
  "Convert a list of inputs to a matrix."
  (let ((length (length inputs)))
    (lisp-matrix:make-matrix length 1
			     :implementation :foreign-array
			     :element-type   'lisp-matrix:single-float
			     :initial-contents (make-array (list length 1)
							   :initial-contents (mapcar #'list
										     inputs)))))

(defun sigmoid (matrix))
(defun activate-layer (layer)
  
)

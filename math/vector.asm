; Asmnet Vector Library (double precision SSE)
; System V AMD64 ABI
;   int args : rdi, rsi, rdx, rcx, r8, r9
;   float args: xmm0..xmm7
;   return   : rax (int) / xmm0 (float)
; Callee-saved registers preserved: rbx, rbp, r12-r15

        global  v_lookup
        global  v_dot
        global  v_rowsum
        global  v_cumsum
        global  v_mean
        global  v_variance
        global  v_rms
        global  v_elem_add
        global  v_elem_sub
        global  v_elem_mul
        global  v_elem_div
        global  v_scalar_mul
        global  v_scalar_div
        global  v_scalar_add
        global  v_scalar_sub
        global  v_sort
        global  v_threshold_vec

        section .text

; ------------------------------------------------------------
; double v_lookup(double *vec, int n, int idx)
;   returns vec[idx] (0.0 if out of bounds)
; ------------------------------------------------------------
v_lookup:
        mov     rax, rdx              ; idx
        cmp     rsi, 0                ; n <= 0
        jle     .zero
        cmp     rax, rsi
        jge     .zero
        movsd   xmm0, qword [rdi + rax*8]
        ret
.zero:
        xorpd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; double v_dot(double *A, double *B, int n)
;   returns sum A[i]*B[i]
; ------------------------------------------------------------
v_dot:
        mov     rcx, rdx              ; n
        test    rcx, rcx
        jle     .zero
        xorpd   xmm0, xmm0            ; acc = 0
        xor     r8, r8                ; i
.loop:
        movsd   xmm1, qword [rdi + r8*8]
        mulsd   xmm1, qword [rsi + r8*8]
        addsd   xmm0, xmm1
        inc     r8
        cmp     r8d, ecx
        jl      .loop
        ret
.zero:
        xorpd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; void v_rowsum(double *A, int rows, int cols, double *out)
;   out[i] = sum of row i
; ------------------------------------------------------------
v_rowsum:
        mov     r10, rcx              ; out
        mov     r11, rdx              ; cols
        xor     r8, r8                ; i
.i_loop:
        cmp     r8d, esi              ; i < rows
        jge     .done
        xorpd   xmm0, xmm0            ; sum = 0
        xor     r9, r9                ; j
.j_loop:
        cmp     r9d, r11d
        jge     .store
        mov     rax, r8
        imul    rax, r11
        add     rax, r9
        shl     rax, 3
        movsd   xmm1, qword [rdi + rax]
        addsd   xmm0, xmm1
        inc     r9
        jmp     .j_loop
.store:
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .i_loop
.done:
        ret

; ------------------------------------------------------------
; void v_cumsum(double *vec, int n, double *out)
;   out[i] = sum_{k=0..i} vec[k]
; ------------------------------------------------------------
v_cumsum:
        mov     r10, rdx              ; out
        test    rsi, rsi              ; n
        jle     .done
        movsd   xmm0, qword [rdi]     ; running = vec[0]
        movsd   [r10], xmm0
        xor     r8, r8                ; i = 0
.loop:
        inc     r8
        cmp     r8d, esi
        jge     .done
        movsd   xmm1, qword [rdi + r8*8]
        addsd   xmm0, xmm1
        movsd   qword [r10 + r8*8], xmm0
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; double v_mean(double *vec, int n)
; ------------------------------------------------------------
v_mean:
        ; do manually
        test    rsi, rsi
        jle     .zero
        xorpd   xmm0, xmm0            ; sum
        xor     r8, r8
.loop:
        addsd   xmm0, qword [rdi + r8*8]
        inc     r8
        cmp     r8d, esi
        jl      .loop
        cvtsi2sd xmm1, rsi            ; n
        divsd   xmm0, xmm1
        ret
.zero:
        xorpd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; double v_variance(double *vec, int n)
;   population variance: mean of (x - mean)^2
; ------------------------------------------------------------
v_variance:
        test    rsi, rsi
        jle     .zero
        ; mean
        push    rbx
        sub     rsp, 16
        movsd   [rsp], xmm0           ; save vec ptr? xmm0 is vec ptr (int reg rdi actually). 
        ; vec ptr is in rdi, n in esi. xmm0 is free.
        xorpd   xmm0, xmm0            ; sum
        xor     r8, r8
.mean_loop:
        addsd   xmm0, qword [rdi + r8*8]
        inc     r8
        cmp     r8d, esi
        jl      .mean_loop
        cvtsi2sd xmm1, rsi
        divsd   xmm0, xmm1            ; xmm0 = mean
        movsd   [rsp + 8], xmm0       ; mean
        ; sum of squares
        xorpd   xmm0, xmm0
        xor     r8, r8
.ssq_loop:
        movsd   xmm1, qword [rdi + r8*8]
        subsd   xmm1, [rsp + 8]       ; x - mean
        mulsd   xmm1, xmm1            ; square
        addsd   xmm0, xmm1
        inc     r8
        cmp     r8d, esi
        jl      .ssq_loop
        cvtsi2sd xmm1, rsi
        divsd   xmm0, xmm1            ; variance
        add     rsp, 16
        pop     rbx
        ret
.zero:
        xorpd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; double v_rms(double *vec, int n)
;   sqrt(mean of x^2)
; ------------------------------------------------------------
v_rms:
        test    rsi, rsi
        jle     .zero
        xorpd   xmm0, xmm0            ; sum of squares
        xor     r8, r8
.loop:
        movsd   xmm1, qword [rdi + r8*8]
        mulsd   xmm1, xmm1
        addsd   xmm0, xmm1
        inc     r8
        cmp     r8d, esi
        jl      .loop
        cvtsi2sd xmm1, rsi
        divsd   xmm0, xmm1            ; mean square
        sqrtsd  xmm0, xmm0
        ret
.zero:
        xorpd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; void v_elem_add(double *A, double *B, int n, double *out)
; ------------------------------------------------------------
v_elem_add:
        mov     r10, rcx              ; out
        xor     r8, r8
.loop:
        cmp     r8d, edx              ; n
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        addsd   xmm0, qword [rsi + r8*8]
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_elem_sub(double *A, double *B, int n, double *out)
; ------------------------------------------------------------
v_elem_sub:
        mov     r10, rcx              ; out
        xor     r8, r8
.loop:
        cmp     r8d, edx              ; n
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        subsd   xmm0, qword [rsi + r8*8]
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_elem_mul(double *A, double *B, int n, double *out)  (Hadamard)
; ------------------------------------------------------------
v_elem_mul:
        mov     r10, rcx              ; out
        xor     r8, r8
.loop:
        cmp     r8d, edx              ; n
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        mulsd   xmm0, qword [rsi + r8*8]
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_elem_div(double *A, double *B, int n, double *out)
; ------------------------------------------------------------
v_elem_div:
        mov     r10, rcx              ; out
        xor     r8, r8
.loop:
        cmp     r8d, edx              ; n
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        divsd   xmm0, qword [rsi + r8*8]
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_scalar_mul(double *vec, int n, double scalar, double *out)
;   Args: rdi=vec, esi=n, xmm0=scalar, rdx=out
; ------------------------------------------------------------
v_scalar_mul:
        mov     r10, rdx              ; out
        movsd   xmm1, xmm0            ; scalar
        xor     r8, r8
.loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        mulsd   xmm0, xmm1
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_scalar_div(double *vec, int n, double scalar, double *out)
; ------------------------------------------------------------
v_scalar_div:
        mov     r10, rdx              ; out
        movsd   xmm1, xmm0            ; scalar
        xor     r8, r8
.loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        divsd   xmm0, xmm1
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_scalar_add(double *vec, int n, double scalar, double *out)
; ------------------------------------------------------------
v_scalar_add:
        mov     r10, rdx              ; out
        movsd   xmm1, xmm0            ; scalar
        xor     r8, r8
.loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        addsd   xmm0, xmm1
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_scalar_sub(double *vec, int n, double scalar, double *out)
; ------------------------------------------------------------
v_scalar_sub:
        mov     r10, rdx              ; out
        movsd   xmm1, xmm0            ; scalar
        xor     r8, r8
.loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        subsd   xmm0, xmm1
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

; ------------------------------------------------------------
; void v_sort(double *vec, int n)  (insertion sort, ascending, in place)
; ------------------------------------------------------------
v_sort:
        test    rsi, rsi
        jle     .done
        mov     r10, rsi              ; n
        dec     r10                   ; n-1
        xor     r8, r8                ; i = 1
.i_loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]   ; key
        mov     r9, r8
        dec     r9                    ; j = i-1
.j_loop:
        test    r9, r9
        jl      .store
        movsd   xmm1, qword [rdi + r9*8]
        comisd  xmm1, xmm0            ; compare vec[j] vs key
        jbe     .store                ; if vec[j] <= key, stop
        movsd   qword [rdi + r9*8 + 8], xmm1   ; vec[j+1] = vec[j]
        dec     r9
        jmp     .j_loop
.store:
        movsd   qword [rdi + r9*8 + 8], xmm0   ; vec[j+1] = key
        inc     r8
        jmp     .i_loop
.done:
        ret

; ------------------------------------------------------------
; void v_threshold_vec(double *vec, int n, double thresh, double *out)
;   out[i] = 1.0 if vec[i] > thresh else 0.0
;   Args: rdi=vec, esi=n, xmm0=thresh, rdx=out
; ------------------------------------------------------------
v_threshold_vec:
        mov     r10, rdx              ; out
        movsd   xmm1, xmm0            ; thresh
        xor     r8, r8
.loop:
        cmp     r8d, esi
        jge     .done
        movsd   xmm0, qword [rdi + r8*8]
        comisd  xmm0, xmm1            ; vec[i] vs thresh
        seta    al                    ; al = 1 if vec[i] > thresh (flags valid)
        movzx   rax, al               ; rax = 0 or 1
        cvtsi2sd xmm0, rax            ; xmm0 = 1.0 or 0.0
        movsd   qword [r10 + r8*8], xmm0
        inc     r8
        jmp     .loop
.done:
        ret

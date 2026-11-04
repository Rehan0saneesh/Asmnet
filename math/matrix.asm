; Asmnet Matrix Library (double precision, SSE)
; System V AMD64 ABI

        global  m_matmul
        global  m_matadd
        global  m_matsub
        global  m_transpose
        global  m_det2
        global  m_det3

        section .text

; ------------------------------------------------------------
; void m_matmul(double *A, int ra, int ca, double *B, int rb, int cb, double *C)
;   C = A x B  (C is ra x cb, pre-allocated by caller; requires ca == rb)
; ------------------------------------------------------------
m_matmul:
        mov     r10, qword [rsp + 8]    ; 7th arg: C pointer
        push    r12
        push    r13
        push    r14
        push    r15
        push    rbx
        push    rbp

        mov     r12, rdi        ; A (ptr)
        movzx   r13, esi        ; ra
        movzx   r14, edx        ; ca
        mov     r15, rcx        ; B (ptr)
        movzx   rbx, r8d        ; rb
        movzx   rbp, r9d        ; cb

        ; validate ca == rb
        cmp     r14d, ebx
        jne     .dim_err

        test    r13d, r13d
        jle     .done
        test    rbp, rbp
        jle     .done

        xor     r8, r8          ; i = 0
.loop_i:
        cmp     r8d, r13d
        jge     .done
        xor     r9, r9          ; j = 0
.loop_j:
        cmp     r9d, ebp
        jge     .next_i
        xorpd   xmm0, xmm0      ; accumulator = 0.0
        xor     r11, r11        ; k = 0
.loop_k:
        cmp     r11d, r14d
        jge     .store
        ; A[i*ca + k]
        mov     rax, r8
        imul    rax, r14
        add     rax, r11
        movsd   xmm1, qword [r12 + rax*8]
        ; B[k*cb + j]
        mov     rax, r11
        imul    rax, rbp
        add     rax, r9
        movsd   xmm2, qword [r15 + rax*8]
        mulsd   xmm1, xmm2
        addsd   xmm0, xmm1
        inc     r11
        jmp     .loop_k
.store:
        mov     rax, r8
        imul    rax, rbp
        add     rax, r9
        movsd   qword [r10 + rax*8], xmm0
        inc     r9
        jmp     .loop_j
.next_i:
        inc     r8
        jmp     .loop_i
.dim_err:
        pop     rbp
        pop     rbx
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        ret
.done:
        pop     rbp
        pop     rbx
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        ret

; ------------------------------------------------------------
; void m_matadd(double *A, int r, int c, double *B, double *C)
;   C = A + B
; ------------------------------------------------------------
m_matadd:
        mov     r10, r8         ; 5th arg: C pointer (in r8)
        push    r12
        push    r13
        push    rbx

        mov     r12, rdi        ; A
        movzx   r13, esi        ; r
        movzx   rbx, edx        ; c
        mov     r14, rcx        ; B

        test    r13d, r13d
        jle     .madd_done
        test    ebx, ebx
        jle     .madd_done
        xor     r8, r8          ; i
.madd_i:
        cmp     r8d, r13d
        jge     .madd_done
        xor     r9, r9          ; j
.madd_j:
        cmp     r9d, ebx
        jge     .madd_next_i
        mov     rax, r8
        imul    rax, rbx
        add     rax, r9
        movsd   xmm0, qword [r12 + rax*8]
        addsd   xmm0, qword [r14 + rax*8]
        movsd   qword [r10 + rax*8], xmm0
        inc     r9
        jmp     .madd_j
.madd_next_i:
        inc     r8
        jmp     .madd_i
.madd_done:
        pop     rbx
        pop     r13
        pop     r12
        ret

; ------------------------------------------------------------
; void m_matsub(double *A, int r, int c, double *B, double *C)
;   C = A - B
; ------------------------------------------------------------
m_matsub:
        mov     r10, r8         ; 5th arg: C pointer (in r8)
        push    r12
        push    r13
        push    rbx

        mov     r12, rdi        ; A
        movzx   r13, esi        ; r
        movzx   rbx, edx        ; c
        mov     r14, rcx        ; B

        test    r13d, r13d
        jle     .msub_done
        test    ebx, ebx
        jle     .msub_done
        xor     r8, r8
.msub_i:
        cmp     r8d, r13d
        jge     .msub_done
        xor     r9, r9
.msub_j:
        cmp     r9d, ebx
        jge     .msub_next_i
        mov     rax, r8
        imul    rax, rbx
        add     rax, r9
        movsd   xmm0, qword [r12 + rax*8]
        subsd   xmm0, qword [r14 + rax*8]
        movsd   qword [r10 + rax*8], xmm0
        inc     r9
        jmp     .msub_j
.msub_next_i:
        inc     r8
        jmp     .msub_i
.msub_done:
        pop     rbx
        pop     r13
        pop     r12
        ret

; ------------------------------------------------------------
; void m_transpose(double *A, int r, int c, double *C)
;   C (c x r) = A^T
; ------------------------------------------------------------
m_transpose:
        mov     r10, rcx        ; 4th arg: C pointer (in rcx)
        push    r12
        push    r13
        push    rbx

        mov     r12, rdi        ; A
        movzx   r13, esi        ; r
        movzx   rbx, edx        ; c

        test    r13d, r13d
        jle     .t_done
        test    ebx, ebx
        jle     .t_done
        xor     r8, r8          ; i
.t_i:
        cmp     r8d, r13d
        jge     .t_done
        xor     r9, r9          ; j
.t_j:
        cmp     r9d, ebx
        jge     .t_next_i
        mov     rax, r8
        imul    rax, rbx
        add     rax, r9
        movsd   xmm0, qword [r12 + rax*8]
        ; C[j*r + i]
        mov     rax, r9
        imul    rax, r13
        add     rax, r8
        movsd   qword [r10 + rax*8], xmm0
        inc     r9
        jmp     .t_j
.t_next_i:
        inc     r8
        jmp     .t_i
.t_done:
        pop     rbx
        pop     r13
        pop     r12
        ret

; ------------------------------------------------------------
; double m_det2(double a, double b, double c, double d)
;   returns a*d - b*c
; ------------------------------------------------------------
m_det2:
        mulsd   xmm0, xmm3      ; a*d
        mulsd   xmm1, xmm2      ; b*c
        subsd   xmm0, xmm1
        ret

; ------------------------------------------------------------
; double m_det3(double a,b,c, d,e,f, g,h,i)
;   |a b c|
;   |d e f|
;   |g h i|
; ------------------------------------------------------------
m_det3:
        movsd   xmm15, qword [rsp + 8]    ; i (9th arg, on stack at entry)
        sub     rsp, 72                   ; 9 slots, rsp 16-aligned; no calls

        movsd   [rsp + 0], xmm0           ; a
        movsd   [rsp + 8], xmm1           ; b
        movsd   [rsp + 16], xmm2          ; c
        movsd   [rsp + 24], xmm3          ; d
        movsd   [rsp + 32], xmm4          ; e
        movsd   [rsp + 40], xmm5          ; f
        movsd   [rsp + 48], xmm6          ; g
        movsd   [rsp + 56], xmm7          ; h
        movsd   [rsp + 64], xmm15         ; i

        ; result = a*(e*i - f*h)
        movsd   xmm0, [rsp + 32]          ; e
        mulsd   xmm0, [rsp + 64]          ; * i
        movsd   xmm1, [rsp + 40]          ; f
        mulsd   xmm1, [rsp + 56]          ; * h
        subsd   xmm0, xmm1                ; ei - fh
        movsd   xmm1, [rsp + 0]           ; a
        mulsd   xmm0, xmm1                ; a*(ei - fh)

        ; result -= b*(d*i - f*g)
        movsd   xmm1, [rsp + 24]          ; d
        mulsd   xmm1, [rsp + 64]          ; * i
        movsd   xmm2, [rsp + 40]          ; f
        mulsd   xmm2, [rsp + 48]          ; * g
        subsd   xmm1, xmm2                ; di - fg
        movsd   xmm2, [rsp + 8]           ; b
        mulsd   xmm1, xmm2                ; b*(di - fg)
        subsd   xmm0, xmm1

        ; result += c*(d*h - e*g)
        movsd   xmm1, [rsp + 24]          ; d
        mulsd   xmm1, [rsp + 56]          ; * h
        movsd   xmm2, [rsp + 32]          ; e
        mulsd   xmm2, [rsp + 48]          ; * g
        subsd   xmm1, xmm2                ; dh - eg
        movsd   xmm2, [rsp + 16]          ; c
        mulsd   xmm1, xmm2                ; c*(dh - eg)
        addsd   xmm0, xmm1

        add     rsp, 72
        ret

; Asmnet Calculus Library (numerical, double precision SSE)
; System V AMD64 ABI
; rbx/r12 are callee-saved and hold the function pointer and n across calls.
; Live XMM values are spilled to stack slots around each user call.

        global  m_integrate
        global  m_derivative

        section .rodata
half:     dq 0.5

        section .text

; ------------------------------------------------------------
; double m_integrate(double (*f)(double), double a, double b, int n)
;   Trapezoidal rule: h=(b-a)/n; integral = h*(0.5*f(a) + sum_{1..n-1} f(a+kh) + 0.5*f(b))
;   Args: rdi=f, xmm0=a, xmm1=b, edx=n
; ------------------------------------------------------------
m_integrate:
        push    rbp
        mov     rbp, rsp
        push    rbx
        push    r12
        sub     rsp, 48               ; 6 slots: a,b,h,sum,x,tmp ; rsp 16-aligned

        mov     rbx, rdi              ; rbx = f (callee-saved)
        mov     r12d, esi             ; r12 = n  (2nd INTEGER arg -> esi)

        movsd   [rbp - 24], xmm0      ; slot a
        movsd   [rbp - 32], xmm1      ; slot b

        ; h = (b - a) / n   -> [rbp - 40]
        movsd   xmm0, [rbp - 32]
        subsd   xmm0, [rbp - 24]
        cvtsi2sd xmm1, r12d
        divsd   xmm0, xmm1
        movsd   [rbp - 40], xmm0      ; slot h

        ; sum = 0   -> [rbp - 48]
        xorpd   xmm0, xmm0
        movsd   [rbp - 48], xmm0      ; slot sum

        ; sum += 0.5 * f(a)
        movsd   xmm0, [rbp - 24]
        call    rbx
        mulsd   xmm0, qword [rel half]
        addsd   xmm0, [rbp - 48]
        movsd   [rbp - 48], xmm0

        ; sum += 0.5 * f(b)
        movsd   xmm0, [rbp - 32]
        call    rbx
        mulsd   xmm0, qword [rel half]
        addsd   xmm0, [rbp - 48]
        movsd   [rbp - 48], xmm0

        ; loop k = 1 .. n-1
        xor     r13d, r13d            ; r13 = k  (caller-saved, but we restore it each iter from stack? no)
        ; r13 is caller-saved! use stack slot for k instead.
        mov     dword [rbp - 56], 0
.loop:
        mov     r13d, dword [rbp - 56]   ; k
        inc     r13d
        mov     dword [rbp - 56], r13d
        cmp     r13d, r12d
        jge     .endloop
        cvtsi2sd xmm0, r13d
        mulsd   xmm0, [rbp - 40]         ; * h
        addsd   xmm0, [rbp - 24]         ; + a
        call    rbx
        addsd   xmm0, [rbp - 48]         ; + sum
        movsd   [rbp - 48], xmm0
        jmp     .loop
.endloop:
        movsd   xmm0, [rbp - 48]         ; sum
        mulsd   xmm0, [rbp - 40]         ; * h

        add     rsp, 48
        pop     r12
        pop     rbx
        pop     rbp
        ret

; ------------------------------------------------------------
; double m_derivative(double (*f)(double), double x, double h)
;   central difference: (f(x+h) - f(x-h)) / (2h)
;   Args: rdi=f, xmm0=x, xmm1=h
; ------------------------------------------------------------
m_derivative:
        push    rbp
        mov     rbp, rsp
        push    rbx
        sub     rsp, 40               ; 5 slots: x,h,fxh ; rsp 16-aligned

        mov     rbx, rdi              ; rbx = f
        movsd   [rbp - 24], xmm0      ; slot x
        movsd   [rbp - 32], xmm1      ; slot h

        ; f(x + h) -> [rbp - 40]? we only allocated 32. use [rbp-24] reused? no.
        ; slots: x=[rbp-24], h=[rbp-32]. need fxh slot. We sub 32 so [rbp-32]..[rbp-56]? no.
        ; Let me recompute: push rbp(8), push rbx(8), sub rsp 32. rsp=rbp-16-32=rbp-48.
        ; slots: [rbp-24],[rbp-32],[rbp-40],[rbp-48] are all below rbp-16? 
        ; rbp-24 is above rsp (rsp=rbp-48). [rbp-24]..[rbp-48] = 24 bytes = 3 slots. Need 3 (x,h,fxh). OK.
        ; but I already put x at [rbp-24], h at [rbp-32]. fxh at [rbp-40].

        movsd   xmm0, [rbp - 24]      ; x
        addsd   xmm0, [rbp - 32]      ; + h
        call    rbx
        movsd   [rbp - 40], xmm0      ; fxh = f(x+h)

        movsd   xmm0, [rbp - 24]      ; x
        subsd   xmm0, [rbp - 32]      ; - h
        call    rbx                   ; xmm0 = f(x-h)
        movsd   xmm1, [rbp - 40]      ; f(x+h)
        subsd   xmm1, xmm0            ; f(x+h) - f(x-h)
        movsd   xmm0, xmm1

        movsd   xmm1, [rbp - 32]      ; h
        addsd   xmm1, xmm1            ; 2h
        divsd   xmm0, xmm1

        add     rsp, 40
        pop     rbx
        pop     rbp
        ret

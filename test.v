// Name : Hemant Mahale
// Roll No : 313057

`timescale 1ns / 1ps

// 1-bit Full Adder module
module FA(A, B, Cin, Sum, Carry);
    input A, B, Cin;
    output Sum, Carry;

    assign Sum = A ^ B ^ Cin;
    assign Carry = (A & B) | (B & Cin) | (A & Cin);
endmodule

// 4-bit Ripple Carry Adder using 1-bit Full Adders
module BPFA(
    input [3:0] A,
    input [3:0] B,
    input Cin,
    output [3:0] Sum,
    output carry
);
    wire c1, c2, c3;

    // Instantiate 4 Full Adders
    FA FA1(A[0], B[0], Cin, Sum[0], c1);
    FA FA2(A[1], B[1], c1, Sum[1], c2);
    FA FA3(A[2], B[2], c2, Sum[2], c3);
    FA FA4(A[3], B[3], c3, Sum[3], carry);
endmodule

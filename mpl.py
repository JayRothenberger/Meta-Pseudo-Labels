import numpy as np
import torch


def MPL(U, L, Y, student, teacher, student_optimizer, teacher_optimizer, loss=torch.nn.CrossEntropyLoss(), sup_teacher=False):
    SPL = teacher(U) # compute the soft pseudo labels
    PL = torch.tensor([np.random.choice(np.arange(SPL.shape[-1]),None,False, torch.nn.Softmax(-1)(SPL).detach().cpu().numpy()[xi]) for xi in range(SPL.shape[0])]) # sample from the SPL distribution
    
    student_initial_output = student(U)
    student_loss_initial = loss(student_initial_output, PL)
    
    student_optimizer.zero_grad()
    student_loss_initial.backward()
    grads1 = [param.grad.data.detach().clone() for param in student.parameters()]

    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()
    
    student_optimizer.zero_grad()
    
    student_final_output = student(L)
    student_loss_final = loss(student_final_output, Y)
    student_loss_final.backward()
    grads2 = [param.grad.data.detach().clone() for param in student.parameters()]

    student_optimizer.zero_grad()
    
    h = sum([(grad1 * grad2).sum() for grad1, grad2 in zip(grads1, grads2)])

    teacher_loss_mpl = h * loss(SPL, PL)
    teacher_out = teacher(L)
    teacher_loss_sup = loss(teacher_out, Y)

    if sup_teacher:
        teacher_loss = teacher_loss_mpl + teacher_loss_sup
    else:
        teacher_loss = teacher_loss_mpl

    teacher_optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_loss.backward()
    teacher_optimizer.step()

    return student_final_output
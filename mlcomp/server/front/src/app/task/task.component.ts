import {Component, OnInit} from '@angular/core';

import {Task} from '../models';
import {TaskService} from '../task.service';
import {Location} from '@angular/common';
import {Router} from '@angular/router';

@Component({
    selector: 'app-task',
    templateUrl: './task.component.html',
    styleUrls: ['./task.component.css']
})
export class TaskComponent implements OnInit {
    tasks: Task[];

    constructor(private taskService: TaskService, private location: Location, private router: Router) {
    }

    ngOnInit() {
        this.getTasks();
    }

    getTasks(): void {
        this.taskService.getTasks()
            .subscribe(tasks => this.tasks = tasks);
    }

    go_back(): void {
        this.location.back();
    }

    go_forward(): void {
        this.location.forward();
    }

}

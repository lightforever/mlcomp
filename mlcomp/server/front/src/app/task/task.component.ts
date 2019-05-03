import {Component, OnInit} from '@angular/core';

import {Task} from '../models';
import {TaskService} from '../task.service';

@Component({
    selector: 'app-task',
    templateUrl: './task.component.html',
    styleUrls: ['./task.component.css']
})
export class TaskComponent implements OnInit {
    tasks: Task[];

    constructor(private taskService: TaskService) {
    }

    ngOnInit() {
        this.get_tasks();
    }

    get_tasks(): void {
        this.taskService.get_tasks()
            .subscribe(res => this.tasks = res.data);
    }

}

import { Component, OnInit, Input } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';

import { Task }         from '../models';
import { TaskService }  from '../task.service';

@Component({
  selector: 'app-task-detail',
  templateUrl: './task-detail.component.html',
  styleUrls: [ './task-detail.component.css' ]
})
export class TaskDetailComponent implements OnInit {
  @Input() task: Task;

  constructor(
    private route: ActivatedRoute,
    private taskService: TaskService,
    private location: Location
  ) {}

  ngOnInit(): void {
    this.get_task();
  }

  get_task(): void {
    const id = +this.route.snapshot.paramMap.get('id');
    this.taskService.get_task(id)
      .subscribe(task => this.task = task);
  }
}

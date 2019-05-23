import { Component, OnInit } from '@angular/core';
import {Paginator} from "../../paginator";
import {TasksComponent} from "../../task/tasks/tasks.component";
import {ReportService} from "../../report.service";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {MatIconRegistry} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {MessageService} from "../../message.service";

@Component({
  selector: 'app-reports',
  templateUrl: './reports.component.html',
  styleUrls: ['./reports.component.css']
})
export class ReportsComponent  extends Paginator<TasksComponent> {

  constructor(protected service: ReportService, protected location: Location,
                private router: Router, private  route: ActivatedRoute,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                private message_service: MessageService
    ) {
        super(service, location);
    }

  protected _ngOnInit() {

  }

  protected displayed_columns: string[] = ["id", "name", "time", "tasks", "tasks_not_finished"];

}

import {Component, Input, OnInit} from '@angular/core';
import {AuxiliarySupervisor} from "../../models";

@Component({
  selector: 'app-auxiliary-supervisor',
  templateUrl: './supervisor.component.html',
  styleUrls: ['./supervisor.component.css']
})
export class AuxiliarySupervisorComponent{
  @Input() data: AuxiliarySupervisor;
}
